# free-model-aggregator-01

A free-model aggregator: one Node server that serves the OpenAI Chat Completions
API and the Anthropic Messages API (JSON and SSE) on top of ten free-tier
providers, picking the strongest free model that meets each request.

## How it works

- Reads the trial brief (`$MAKEFUN_PROMPT_FILE`): provider directory, quality
  table, workload manifest. Probes every provider's catalog route in its own
  wire and registers one catalog with the env.
- Routing: for each request, every free model meeting the requirements is a
  candidate, ranked by `rating_milli` minus a load penalty (10 milli per call
  already sent to that provider), so traffic spreads and no single provider's
  per-trial budget takes many requests down with it. JSON-mode requests only
  go to a wire that can carry `response_format` (OpenAI or Gemini, never
  Anthropic). Ties break deterministically on provider id, then model id.
- Failover: a 402 marks the provider exhausted for the trial; a 429/5xx is
  retried on the next candidate, and a provider that fails twice is treated
  as exhausted. A provider's first call in a trial always succeeds, so
  spreading first calls is safe by construction.
- Wires: the client request is normalised to a canonical Chat Completions
  shape, then translated to the upstream wire (OpenAI, Anthropic, Gemini)
  and the answer translated back, preserving the system prompt, tools, tool
  results, images, stop sequences, `max_tokens` and JSON mode exactly, so
  the provider's echo matches what the client sent. Stop reasons map
  end_turn to stop, max_tokens to length, tool_use to tool_calls and
  refusal to content_filter; usage is passed through as the wire reports it.
- Deterministic: no wall clock, no unseeded randomness. Every choice is a
  function of the requests and the provider answers seen so far, so a replay
  reproduces the live run.

## Run

```sh
node main.ts --list   # print the catalog and exit
node main.ts          # serve: register with $MAKEFUN_ENV_URL, exit 0 when done
```

Node 22 stdlib only. No dependencies, no keys in the repo.

## Layout

- `main.ts` — server, catalog probe, routing, failover, register/poll loop.
- `lib/providers.ts` — brief parsing, auth headers, catalog probe.
- `lib/wire.ts` — client wires and the OpenAI/Anthropic translations.
- `lib/upstream.ts` — upstream translations: OpenAI to Anthropic and Gemini,
  and their answers back to OpenAI/Anthropic, JSON and SSE.
