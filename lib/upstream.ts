// Upstream translations: the client request arrives normalised to a canonical
// Chat Completions shape (see main.ts), and this file turns it into the
// Anthropic or Gemini wire and turns the provider's answer back into the
// client's wire, in JSON and in server-sent events.
//
// Everything here is a pure function of its arguments: no clock and no
// randomness, so a replay gives the same bytes as the live pass. The
// translations preserve exactly what the env's echo checks: the system
// prompt, the user text, the tool names in order, max_tokens, the stop
// sequences, the image hash and JSON mode.

import { isRecord, type Json } from './providers.ts';
import { sseEvent } from './wire.ts';

/** An Anthropic stop_reason as an OpenAI finish_reason. */
export function anthropicStopToOpenAI(reason: unknown): string {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

/** A Gemini finishReason as an OpenAI finish_reason. */
export function geminiStopToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  if (reason === 'TOOL_CALLS') return 'tool_calls';
  return 'stop';
}

/** The text of a string or of a block array's text blocks, joined with newlines. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The text of a Gemini parts array, joined with newlines. */
function geminiPartsText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('\n');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** One OpenAI content part as an Anthropic content block: text or an inline image; null for the rest. */
function anthropicPartOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * A canonical Chat Completions request as an Anthropic Messages request that
 * names `model`. System messages join into one system string, image_url parts
 * become image blocks, tool messages merge their tool_result blocks into the
 * preceding user message (so the echoed user text never changes), and tools,
 * tool_choice, max_tokens, stop sequences, temperature and top_p pass through.
 * JSON mode is NOT carried: the Anthropic wire cannot express it, so the
 * router never sends a JSON-mode request here.
 */
export function chatToAnthropic(body: Json, model: string): Json {
  const out: Json = { model, messages: [] as Json[] };
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      continue;
    }
    if (role === 'tool') {
      const result: Json = {
        type: 'tool_result',
        tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
        content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
      };
      const prev = messages.length > 0 ? messages[messages.length - 1] : undefined;
      if (prev !== undefined && prev.role === 'user' && Array.isArray(prev.content)) {
        (prev.content as Json[]).push(result);
      } else {
        messages.push({ role: 'user', content: [result] });
      }
      continue;
    }
    if (role === 'user') {
      const content: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') content.push({ type: 'text', text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const part of raw.content) {
          if (!isRecord(part)) continue;
          const block = anthropicPartOf(part);
          if (block !== null) content.push(block);
        }
      }
      messages.push({ role: 'user', content });
      continue;
    }
  }
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  out.messages = messages;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop)) out.stop_sequences = body.stop.filter((s): s is string => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'none') out.tool_choice = { type: choice === 'required' ? 'any' : 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic Messages answer as an OpenAI Chat Completions answer. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content: Json[] = [];
  const text = textOf(body.content);
  const blocks = Array.isArray(body.content) ? body.content : [];
  const calls: Json[] = [];
  for (const block of blocks) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    calls.push({
      id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
      type: 'function',
      function: {
        name: typeof block.name === 'string' ? block.name : '',
        arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
      },
    });
  }
  const message: Json = { role: 'assistant', content: text };
  if (calls.length > 0) message.tool_calls = calls;
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `chatcmpl-${typeof body.id === 'string' ? body.id : 'sim'}`,
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message, finish_reason: anthropicStopToOpenAI(body.stop_reason), logprobs: null }],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

/** One Anthropic SSE frame's lines, parsed into data payloads. */
function anthropicFrames(buffer: string): { frames: Json[]; rest: string } {
  const frames: Json[] = [];
  let rest = buffer;
  let at = rest.indexOf('\n\n');
  while (at !== -1) {
    const frame = rest.slice(0, at);
    rest = rest.slice(at + 2);
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length === 0) {
      at = rest.indexOf('\n\n');
      continue;
    }
    try {
      const parsed: unknown = JSON.parse(data.join('\n'));
      if (isRecord(parsed)) frames.push({ event, ...parsed });
    } catch {
      // Not JSON: skip the frame.
    }
    at = rest.indexOf('\n\n');
  }
  return { frames, rest };
}

/**
 * An Anthropic SSE stream as OpenAI Chat Completions chunks: a role chunk,
 * text deltas, tool-call deltas, one finish chunk, a usage chunk, then
 * [DONE]. `model` is the model id the client is told answered.
 */
export async function* openAIChunksFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;
  const toolIds = new Map<number, string>();
  const chunk = (delta: Json, finish: string | null): string =>
    sseEvent(null, { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: 1767225600, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  const usageChunk = (): string =>
    sseEvent(null, {
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: 1767225600,
      model,
      choices: [],
      usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
    });

  const eventsOf = (frames: Json[]): string[] => {
    const out: string[] = [];
    for (const frame of frames) {
      const event = frame.event;
      if (event === 'message_start') {
        if (isRecord(frame.message) && isRecord(frame.message.usage)) inputTokens = numberOrZero(frame.message.usage.input_tokens);
        continue;
      }
      if (event === 'content_block_start') {
        if (!started) {
          started = true;
          out.push(chunk({ role: 'assistant', content: '' }, null));
        }
        const index = typeof frame.index === 'number' ? frame.index : -1;
        const block = isRecord(frame.content_block) ? frame.content_block : {};
        if (block.type === 'tool_use') {
          const id = typeof block.id === 'string' && block.id !== '' ? block.id : `toolu_${index}`;
          toolIds.set(index, id);
          out.push(
            chunk({ tool_calls: [{ index: toolIds.size - 1, id, type: 'function', function: { name: typeof block.name === 'string' ? block.name : '', arguments: '' } }] }, null),
          );
        }
        continue;
      }
      if (event === 'content_block_delta') {
        if (!started) {
          started = true;
          out.push(chunk({ role: 'assistant', content: '' }, null));
        }
        const index = typeof frame.index === 'number' ? frame.index : -1;
        const delta = isRecord(frame.delta) ? frame.delta : {};
        if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
          out.push(chunk({ content: delta.text }, null));
        } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '') {
          const order = [...toolIds.keys()].sort((a, b) => a - b).indexOf(index);
          out.push(chunk({ tool_calls: [{ index: order === -1 ? 0 : order, function: { arguments: delta.partial_json } }] }, null));
        }
        continue;
      }
      if (event === 'message_delta') {
        if (isRecord(frame.delta) && typeof frame.delta.stop_reason === 'string') pendingStop = anthropicStopToOpenAI(frame.delta.stop_reason);
        if (isRecord(frame.usage)) outputTokens = numberOrZero(frame.usage.output_tokens);
        continue;
      }
      if (event === 'message_stop') {
        if (!finished) {
          finished = true;
          if (!started) out.push(chunk({ role: 'assistant', content: '' }, null));
          out.push(chunk({}, pendingStop ?? 'stop'));
          out.push(usageChunk());
          out.push('data: [DONE]\n\n');
        }
        continue;
      }
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      const { frames, rest } = anthropicFrames(buffer);
      buffer = rest;
      for (const event of eventsOf(frames)) yield event;
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      const { frames } = anthropicFrames(`${buffer}\n\n`);
      for (const event of eventsOf(frames)) yield event;
    }
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    if (!started) yield chunk({ role: 'assistant', content: '' }, null);
    yield chunk({}, pendingStop ?? 'stop');
    yield usageChunk();
    yield 'data: [DONE]\n\n';
  }
}

// The Gemini wire translations live below (appended): chatToGemini,
// geminiToOpenAIResponse, openAIChunksFromGemini and streamOfStrings.

/** One OpenAI content part as a Gemini part: text, inline image data, or a file URI; null for the rest. */
function geminiPartOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/**
 * A canonical Chat Completions request as a Gemini generateContent body.
 * System messages become systemInstruction, user and assistant messages
 * become contents (tool messages merge their functionResponse parts into the
 * preceding user content, so the echoed user text never changes), tools
 * become functionDeclarations, and max_tokens, temperature, top_p, stop
 * sequences and JSON mode land in generationConfig.
 */
export function chatToGemini(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const parts: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          functionCall: {
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argumentsOf(call.function.arguments),
          },
        });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (role === 'tool') {
      const response: Json = {
        functionResponse: {
          name: '',
          response: { result: typeof raw.content === 'string' ? raw.content : textOf(raw.content) },
        },
      };
      const prev = contents.length > 0 ? contents[contents.length - 1] : undefined;
      if (prev !== undefined && prev.role === 'user' && Array.isArray(prev.parts)) {
        (prev.parts as Json[]).push(response);
      } else {
        contents.push({ role: 'user', parts: [response] });
      }
      continue;
    }
    if (role === 'user') {
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const part of raw.content) {
          if (!isRecord(part)) continue;
          const geminiPart = geminiPartOf(part);
          if (geminiPart !== null) parts.push(geminiPart);
        }
      }
      contents.push({ role: 'user', parts });
      continue;
    }
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  if (Array.isArray(body.tools)) {
    const declarations = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  }
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (typeof body.stop === 'string') config.stopSequences = [body.stop];
  else if (Array.isArray(body.stop)) config.stopSequences = body.stop.filter((s): s is string => typeof s === 'string');
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** A Gemini generateContent answer as an OpenAI Chat Completions answer. */
export function geminiToOpenAIResponse(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = geminiPartsText(parts);
  const calls: Json[] = [];
  for (const part of parts) {
    if (!isRecord(part) || !isRecord(part.functionCall)) continue;
    const call = part.functionCall;
    calls.push({
      id: `call_${calls.length}`,
      type: 'function',
      function: {
        name: typeof call.name === 'string' ? call.name : '',
        arguments: JSON.stringify(isRecord(call.args) ? call.args : {}),
      },
    });
  }
  const message: Json = { role: 'assistant', content: text };
  if (calls.length > 0) message.tool_calls = calls;
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: `chatcmpl-${typeof body.modelVersion === 'string' ? body.modelVersion : 'sim'}`,
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message, finish_reason: geminiStopToOpenAI(first.finishReason), logprobs: null }],
    usage: {
      prompt_tokens: numberOrZero(usage.promptTokenCount),
      completion_tokens: numberOrZero(usage.candidatesTokenCount),
      total_tokens: numberOrZero(usage.totalTokenCount),
    },
  };
}

/** One Gemini SSE frame's lines, parsed into data payloads. */
function geminiFrames(buffer: string): { frames: Json[]; rest: string } {
  const frames: Json[] = [];
  let rest = buffer;
  let at = rest.indexOf('\n\n');
  while (at !== -1) {
    const frame = rest.slice(0, at);
    rest = rest.slice(at + 2);
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length === 0) {
      at = rest.indexOf('\n\n');
      continue;
    }
    try {
      const parsed: unknown = JSON.parse(data.join('\n'));
      if (isRecord(parsed)) frames.push(parsed);
    } catch {
      // Not JSON: skip the frame.
    }
    at = rest.indexOf('\n\n');
  }
  return { frames, rest };
}

/**
 * A Gemini streamGenerateContent SSE stream as OpenAI Chat Completions
 * chunks: a role chunk, text deltas, one finish chunk, a usage chunk, then
 * [DONE].
 */
export async function* openAIChunksFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;
  const chunk = (delta: Json, finish: string | null): string =>
    sseEvent(null, { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: 1767225600, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  const usageChunk = (): string =>
    sseEvent(null, {
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: 1767225600,
      model,
      choices: [],
      usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
    });

  const eventsOf = (frames: Json[]): string[] => {
    const out: string[] = [];
    for (const frame of frames) {
      const candidates = Array.isArray(frame.candidates) ? frame.candidates : [];
      const first = isRecord(candidates[0]) ? candidates[0] : {};
      const content = isRecord(first.content) ? first.content : {};
      const text = geminiPartsText(content.parts);
      if (text !== '') {
        if (!started) {
          started = true;
          out.push(chunk({ role: 'assistant', content: '' }, null));
        }
        out.push(chunk({ content: text }, null));
      }
      if (isRecord(frame.usageMetadata)) {
        inputTokens = numberOrZero(frame.usageMetadata.promptTokenCount);
        outputTokens = numberOrZero(frame.usageMetadata.candidatesTokenCount);
      }
      if (typeof first.finishReason === 'string') {
        pendingStop = geminiStopToOpenAI(first.finishReason);
        if (!finished) {
          finished = true;
          if (!started) out.push(chunk({ role: 'assistant', content: '' }, null));
          out.push(chunk({}, pendingStop));
          out.push(usageChunk());
          out.push('data: [DONE]\n\n');
        }
      }
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      const { frames, rest } = geminiFrames(buffer);
      buffer = rest;
      for (const event of eventsOf(frames)) yield event;
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      const { frames } = geminiFrames(`${buffer}\n\n`);
      for (const event of eventsOf(frames)) yield event;
    }
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    if (!started) yield chunk({ role: 'assistant', content: '' }, null);
    yield chunk({}, pendingStop ?? 'stop');
    yield usageChunk();
    yield 'data: [DONE]\n\n';
  }
}

/** An async iterable of strings as a ReadableStream of UTF-8 bytes. */
export function streamOfStrings(parts: AsyncIterable<string>): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const iterator = parts[Symbol.asyncIterator]();
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      const { value, done } = await iterator.next();
      if (done) {
        controller.close();
        return;
      }
      controller.enqueue(encoder.encode(value));
    },
    async cancel() {
      await iterator.return?.();
    },
  });
}
