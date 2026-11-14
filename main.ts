// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the trial brief, probes each provider's catalog route in
// its own wire, registers one catalog with the env, and serves the client's
// OpenAI and Anthropic requests (JSON and SSE) from the strongest free model
// that meets each request, failing over when a provider refuses.
//
// ROUTING (quality is 60% of the score, served 15%)
//   Every free model meeting a request's requirements is a candidate, ranked
//   by rating_milli minus a load penalty of 10 milli per call already sent
//   to its provider, so traffic spreads across providers instead of draining
//   one provider's per-trial budget. Ties break on provider id, then model
//   id: fully deterministic. A JSON-mode request only goes to a wire that
//   can carry response_format (OpenAI or Gemini, never Anthropic).
//
// FAILOVER (10% of the score)
//   A 402 marks the provider exhausted for the trial; a 429 or 5xx is
//   retried on the next candidate, and a provider that fails twice is
//   treated as exhausted. A failed call still counts against the provider's
//   trial counter (the env counts every call that names a free model), so
//   the load penalty already accounts for it. One provider serves one call
//   at a time and the client sends one request at a time, so this server
//   never makes concurrent provider calls.
//
// WIRES
//   A client request is normalised to a canonical Chat Completions shape,
//   translated to the upstream wire (OpenAI, Anthropic or Gemini) by
//   lib/upstream.ts, and the provider's answer is translated back into the
//   client's wire, preserving the system prompt, tools, tool results,
//   images, stop sequences, max_tokens and JSON mode exactly, so the
//   provider's echo matches what the client sent. Streaming answers are
//   only started once a provider answers 2xx, so a failed first attempt
//   never leaks a partial stream to the client.
//
// DETERMINISM (a replay must reproduce the live run byte for byte)
//   No wall clock, no unseeded randomness. Every choice is a function of
//   the requests and the provider answers seen so far.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        Anthropic <-> Chat Completions translation and the SSE writers.
//   lib/upstream.ts    the Anthropic and Gemini upstream wires, both directions, JSON and SSE.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type RatingRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicToOpenAIResponse,
  chatToAnthropic,
  chatToGemini,
  geminiToOpenAIResponse,
  openAIChunksFromAnthropic,
  openAIChunksFromGemini,
  streamOfStrings,
} from './lib/upstream.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Rating milli lost per call already sent to a provider: spreads the load. */
const LOAD_PENALTY_MILLI = 10;
/** Failed calls after which a provider counts as exhausted for the trial. */
const EXHAUST_AFTER_FAILURES = 2;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

/** Calls sent to each provider so far (the env counts every call that names a free model). */
const usedCalls = new Map<string, number>();
/** Failed calls per provider so far (402, 429, 5xx, or a transport error). */
const failedCalls = new Map<string, number>();
/** Providers treated as exhausted for the rest of the trial. */
const exhausted = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  provider: ProviderDirectoryEntry;
  probe: Probe;
  model: ProbedModel;
  row: RatingRow | null;
  rating: number;
}

/** Every free model the probes listed, with its rating row: the routing pool. */
function allCandidates(): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ provider: probe.provider, probe, model, row, rating: row?.rating_milli ?? 0 });
    }
  }
  return out;
}

let candidatePool: Candidate[] = [];

/**
 * The candidates for one request, best first: free models meeting the
 * requirements, ranked by rating minus the load penalty for calls already
 * sent to their provider, then by rating, provider id and model id.
 * Exhausted providers sort last (unless every candidate is exhausted, in
 * which case they are tried anyway, best first). A JSON-mode request only
 * goes to a wire that can carry response_format.
 */
function rankCandidates(requires: Requirement): Candidate[] {
  const list = candidatePool.filter((c) => meetsRequirements(requires, capabilitiesOf(c.model, c.row)));
  const json = requires.json === true;
  const usable = json ? list.filter((c) => c.provider.wire !== 'anthropic') : list;
  const anyUsable = usable.some((c) => !exhausted.has(c.provider.id));
  const score = (c: Candidate): number => c.rating - LOAD_PENALTY_MILLI * (usedCalls.get(c.provider.id) ?? 0);
  return [...usable].sort((a, b) => {
    const aEx = exhausted.has(a.provider.id);
    const bEx = exhausted.has(b.provider.id);
    if (anyUsable && aEx !== bEx) return aEx ? 1 : -1;
    const byScore = score(b) - score(a);
    if (byScore !== 0) return byScore;
    if (a.rating !== b.rating) return b.rating - a.rating;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
}

function noteCall(providerId: string): void {
  usedCalls.set(providerId, (usedCalls.get(providerId) ?? 0) + 1);
}

function noteFailure(providerId: string, status: number): void {
  const failures = (failedCalls.get(providerId) ?? 0) + 1;
  failedCalls.set(providerId, failures);
  // A 402 is a spent budget: permanent for the trial. Two failures of any
  // kind mean the provider is out or on its outage, so stop spending calls.
  if (status === 402 || failures >= EXHAUST_AFTER_FAILURES) exhausted.add(providerId);
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The client body normalised to a canonical Chat Completions request naming `model`. */
function canonicalBody(clientApi: 'openai' | 'anthropic', body: Json, model: string): Json {
  if (clientApi === 'openai') {
    const out: Json = { ...body, model };
    if (out.stream === true) out.stream_options = { include_usage: true };
    return out;
  }
  return anthropicToOpenAI(body, model);
}

/** One upstream request body for a candidate's wire. */
function upstreamBody(clientApi: 'openai' | 'anthropic', body: Json, candidate: Candidate): Json {
  const canonical = canonicalBody(clientApi, body, candidate.model.id);
  if (candidate.provider.wire === 'openai') return canonical;
  if (candidate.provider.wire === 'anthropic') return chatToAnthropic(canonical, candidate.model.id);
  return chatToGemini(canonical, candidate.model.id);
}

/** POST one request to a candidate provider, in its own wire. */
async function callUpstream(candidate: Candidate, payload: Json, stream: boolean): Promise<Response> {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  let url: string;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
  } else {
    const action = stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}${stream ? '?alt=sse' : ''}`;
  }
  noteCall(provider.id);
  return fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Write an async generator of SSE strings to the client, then end the response. */
async function pipeChunks(chunks: AsyncGenerator<string>, res: ServerResponse): Promise<void> {
  for await (const event of chunks) res.write(event);
  res.end();
}

/** Deliver one successful upstream answer to the client, in the client's wire. */
async function deliver(
  clientApi: 'openai' | 'anthropic',
  candidate: Candidate,
  upstream: Response,
  body: Json,
  res: ServerResponse,
): Promise<void> {
  const stream = body.stream === true;
  const wire = candidate.provider.wire;
  const model = candidate.model.id;
  const requested = clientApi === 'anthropic' && typeof body.model === 'string' && body.model !== '' ? body.model : model;
  if (clientApi === 'openai') {
    if (wire === 'openai') {
      if (stream && upstream.body !== null) {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
        await pipeStream(upstream.body, res);
        return;
      }
      const text = await upstream.text();
      res.writeHead(200, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
      res.end(text);
      return;
    }
    if (stream && upstream.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      const chunks = wire === 'anthropic' ? openAIChunksFromAnthropic(upstream.body, model) : openAIChunksFromGemini(upstream.body, model);
      await pipeChunks(chunks, res);
      return;
    }
    const answer: unknown = await upstream.json();
    const converted = wire === 'anthropic' ? anthropicToOpenAIResponse(isRecord(answer) ? answer : {}, model) : geminiToOpenAIResponse(isRecord(answer) ? answer : {}, model);
    sendJson(res, 200, converted);
    return;
  }
  // Anthropic client.
  if (wire === 'anthropic') {
    if (stream && upstream.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      await pipeStream(upstream.body, res);
      return;
    }
    const text = await upstream.text();
    res.writeHead(200, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(text);
    return;
  }
  if (wire === 'openai') {
    if (stream && upstream.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      for await (const event of anthropicEventsFromOpenAI(upstream.body, requested)) res.write(event);
      res.end();
      return;
    }
    const answer: unknown = await upstream.json();
    sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, requested));
    return;
  }
  // Anthropic client, Gemini upstream: Gemini -> OpenAI -> Anthropic.
  if (stream && upstream.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    const openAIStream = streamOfStrings(openAIChunksFromGemini(upstream.body, model));
    for await (const event of anthropicEventsFromOpenAI(openAIStream, requested)) res.write(event);
    res.end();
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIToAnthropic(geminiToOpenAIResponse(isRecord(answer) ? answer : {}, model), requested));
}

/**
 * Serve one client request: try the ranked candidates in order until one
 * answers 2xx, failing over on 402, 429, 5xx or a transport error. A
 * streaming answer only starts once a provider has answered, so a failed
 * attempt never leaks a partial stream.
 */
async function serve(clientApi: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const ranked = rankCandidates(requirementsFor(body));
  for (const candidate of ranked) {
    const payload = upstreamBody(clientApi, body, candidate);
    let upstream: Response;
    try {
      upstream = await callUpstream(candidate, payload, body.stream === true);
    } catch (err) {
      log(`call to ${candidate.provider.id}/${candidate.model.id} failed: ${(err as Error).message}`);
      noteFailure(candidate.provider.id, 0);
      continue;
    }
    if (upstream.ok) {
      await deliver(clientApi, candidate, upstream, body, res);
      return;
    }
    try {
      await upstream.text();
    } catch {
      // The error body is unreadable: the status is what matters.
    }
    log(`call to ${candidate.provider.id}/${candidate.model.id} refused HTTP ${upstream.status}`);
    noteFailure(candidate.provider.id, upstream.status);
  }
  sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  candidatePool = allCandidates();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
