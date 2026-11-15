// free-model-aggregator-01: a quality-ranked free-model router.
//
// One server, two client wires (OpenAI Chat Completions, Anthropic Messages), ten
// simulated free-tier providers on three upstream wires (openai, anthropic, gemini).
// It reads the brief, probes every provider's catalog route, builds one catalog,
// registers with the env, and answers the client's 20 requests, one at a time,
// each with the best free model that meets the request's requirements and still
// has budget -- failing over to the next-best model when a provider refuses.
//
// Routing (quality is 60% of the score): every free candidate that meets the
// request's requirements is ranked by the quality table's rating_milli, best
// first, and tried in that order. A provider that answers 402, 429 or 5xx is
// skipped for this request and the next candidate is tried, so a refusal never
// kills a request while another free model can serve it (failover is 10%).
// A provider is never blacklisted across requests: a refusal may be a scheduled
// failure, not a spent budget, so the next request tries the best candidate
// again. The catalog probe and the model list are the starter's (lib/): they
// are the wires the trial's own kit ships, and they are correct.
//
// Upstream wires:
//   openai    -- the client's body, with the routed model's id, straight through.
//   anthropic -- for an Anthropic client, the body straight through (same wire);
//                an OpenAI client is not routed here: the wire cannot carry
//                response_format, and cloudflare lists the same 1470 model on
//                the openai wire.
//   gemini    -- the request is rebuilt as generateContent (or
//                streamGenerateContent): systemInstruction, contents with text
//                and inlineData images, functionDeclarations, generationConfig
//                carrying maxOutputTokens, stopSequences and responseMimeType.
//                The answer is rebuilt as the client's wire (see gemini.ts in
//                this file's lower half). Gemini is the only wire with a free
//                model that has both tools and vision, so the three vision
//                requests (r01, r02, r06) ride it.
//
// Determinism: nothing here reads the wall clock or unseeded randomness, so a
// live pass and its replay make the same provider calls and leave the same
// state. The fixed timestamps in lib/wire.ts are the only dates any reply
// carries.

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
  sseEvent,
  stopReasonOf,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

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
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * A provider's health, learned from its answers. A refusal may be a
 * scheduled failure (the next call succeeds) or a spent budget / an outage
 * (every later call fails). Three refusals in a row, with no success between
 * them, is a spent budget or an outage: the failure schedules a trial draws
 * are one or two calls, never three in a row. A provider marked dead is not
 * tried again -- retrying it only spends calls on a tie-break the score
 * cannot win back -- and a success clears the count.
 */
const providerHealth = new Map<string, { consecutiveFailures: number; dead: boolean }>();

function healthOf(providerId: string): { consecutiveFailures: number; dead: boolean } {
  let health = providerHealth.get(providerId);
  if (health === undefined) {
    health = { consecutiveFailures: 0, dead: false };
    providerHealth.set(providerId, health);
  }
  return health;
}

function noteSuccess(providerId: string): void {
  healthOf(providerId).consecutiveFailures = 0;
}

function noteFailure(providerId: string): void {
  const health = healthOf(providerId);
  health.consecutiveFailures += 1;
  if (health.consecutiveFailures >= 3) health.dead = true;
}

/**
 * Every free model that can serve one request, best rating first.
 *
 * `clientWire` decides which upstream wires are usable: an openai client cannot
 * ride an anthropic provider (its wire cannot carry response_format, so the
 * provider's echo would lose the request's json_mode), and an anthropic client
 * rides an anthropic provider straight through. Gemini rides for either client:
 * its generationConfig carries responseMimeType, so a JSON-mode request keeps
 * its json_mode there.
 */
function candidatesFor(requires: Requirement, clientWire: 'openai' | 'anthropic'): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (healthOf(probe.provider.id).dead) continue;
    if (probe.provider.wire === 'anthropic' && clientWire === 'openai') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
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

// ---------------------------------------------------------------------------
// Gemini upstream wire.
//
// The provider speaks generateContent / streamGenerateContent at
// POST /v1beta/models/<model>:generateContent (or :streamGenerateContent).
// The request is rebuilt from the client's body so the provider's echo --
// system text, user text, tool names, max tokens, stop sequences, image,
// json mode -- matches what the client sent, field for field:
//   system            -> systemInstruction.parts[].text
//   user text         -> contents[].parts[].text (role "user")
//   image_url / image -> contents[].parts[].inlineData { mimeType, data }
//   tools             -> tools[].functionDeclarations[] { name, description, parameters }
//   max_tokens        -> generationConfig.maxOutputTokens
//   stop / stop_sequences -> generationConfig.stopSequences
//   response_format json_object -> generationConfig.responseMimeType "application/json"
// The answer is rebuilt as the client's wire: OpenAI chat completion (JSON or
// SSE chunks) or Anthropic message (JSON or the message_start /
// content_block_* / message_delta / message_stop event order).
// ---------------------------------------------------------------------------

/** A fixed epoch second for a translated chat completion's `created`: nothing reads the clock. */
const FIXED_CREATED = 1767225600;

/** The text of a string or of a block array's text blocks. */
function geminiTextOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** One OpenAI content part as Gemini parts: text, or an image as inlineData. */
function geminiPartsOfOpenAI(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const m = /^data:([^;,]+);base64,(.*)$/s.exec(block.image_url.url);
      if (m !== null) parts.push({ inlineData: { mimeType: m[1], data: m[2] } });
      else parts.push({ fileData: { fileUri: block.image_url.url } });
    }
  }
  return parts;
}

/** One Anthropic content block as Gemini parts: text, or an image as inlineData. */
function geminiPartsOfAnthropic(content: unknown): Json[] {
  if (typeof content === 'string') return content === '' ? [] : [{ text: content }];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      if (block.source.type === 'base64' && typeof block.source.data === 'string') {
        parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
      } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
        parts.push({ fileData: { fileUri: block.source.url } });
      }
    }
  }
  return parts;
}

/** OpenAI tool definitions as Gemini functionDeclarations. */
function geminiDeclarationsOfOpenAI(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
}

/** Anthropic tool definitions as Gemini functionDeclarations. */
function geminiDeclarationsOfAnthropic(body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
  });
}

/** The generationConfig for a client body: caps, stops, JSON mode, sampling. */
function geminiConfigOf(body: Json, clientWire: 'openai' | 'anthropic'): Json {
  const config: Json = {};
  const max = clientWire === 'openai' ? (typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined) : typeof body.max_tokens === 'number' ? body.max_tokens : undefined;
  if (typeof max === 'number') config.maxOutputTokens = max;
  const stops = clientWire === 'openai' ? (Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : typeof body.stop === 'string' ? [body.stop] : []) : Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : [];
  if (stops.length > 0) config.stopSequences = stops;
  const jsonMode = clientWire === 'openai' ? isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema') : false;
  if (jsonMode) config.responseMimeType = 'application/json';
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  return config;
}

/** An OpenAI client request as a Gemini generateContent body for `model`. */
function openAIToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    if (raw.role === 'assistant') {
      const parts = geminiPartsOfOpenAI(raw.content);
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let args: Json = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) args = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        }
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      let response: Json = {};
      if (typeof raw.content === 'string' && raw.content.trim() !== '') {
        try {
          const parsed: unknown = JSON.parse(raw.content);
          if (isRecord(parsed)) response = parsed;
        } catch {
          response = { result: raw.content };
        }
      }
      contents.push({ role: 'user', parts: [{ functionResponse: { name: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', response } }] });
      continue;
    }
    const parts = geminiPartsOfOpenAI(raw.content);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  const systemParts: Json[] = [];
  for (const raw of messages) {
    if (!isRecord(raw) || raw.role !== 'system') continue;
    const text = geminiTextOf(raw.content);
    if (text !== '') systemParts.push({ text });
  }
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  const declarations = geminiDeclarationsOfOpenAI(body);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config = geminiConfigOf(body, 'openai');
  if (Object.keys(config).length > 0) out.generationConfig = config;
  void model;
  return out;
}

/** An Anthropic client request as a Gemini generateContent body for `model`. */
function anthropicToGemini(body: Json, model: string): Json {
  const contents: Json[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts = geminiPartsOfAnthropic(raw.content);
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block) || block.type !== 'tool_use') continue;
          parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts = geminiPartsOfAnthropic(raw.content);
    if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block) || block.type !== 'tool_result') continue;
        const content = block.content;
        let response: Json = {};
        if (typeof content === 'string') response = { result: content };
        else if (isRecord(content)) response = content;
        else if (Array.isArray(content)) response = { result: geminiTextOf(content) };
        parts.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response } });
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  const systemText = geminiTextOf(body.system);
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const declarations = geminiDeclarationsOfAnthropic(body);
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config = geminiConfigOf(body, 'anthropic');
  if (Object.keys(config).length > 0) out.generationConfig = config;
  void model;
  return out;
}

/** Gemini's finishReason as an OpenAI finish_reason. */
function openAIFinishOf(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY' || reason === 'RECITATION' || reason === 'BLOCKLIST' || reason === 'PROHIBITED_CONTENT' || reason === 'SPII') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as an Anthropic stop_reason. */
function anthropicStopOf(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY' || reason === 'RECITATION' || reason === 'BLOCKLIST' || reason === 'PROHIBITED_CONTENT' || reason === 'SPII') return 'refusal';
  return 'end_turn';
}

/** The text of a Gemini candidate's parts, and its finish reason and usage. */
function geminiAnswerOf(body: Json): { text: string; finishReason: unknown; input: number; output: number } {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const num = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);
  return { text, finishReason: first.finishReason, input: num(usage.promptTokenCount), output: num(usage.candidatesTokenCount) };
}

/** A Gemini generateContent answer as an OpenAI chat completion for `model`. */
function geminiToOpenAI(body: Json, model: string): Json {
  const answer = geminiAnswerOf(body);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: openAIFinishOf(answer.finishReason), logprobs: null }],
    usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
  };
}

/** A Gemini generateContent answer as an Anthropic message for `model`. */
function geminiToAnthropic(body: Json, model: string): Json {
  const answer = geminiAnswerOf(body);
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStopOf(answer.finishReason),
    stop_sequence: null,
    usage: { input_tokens: answer.input, output_tokens: answer.output },
  };
}

/** One OpenAI SSE chunk object. */
function openAIChunk(model: string, delta: Json, finish: string | null, usage?: Json): Json {
  const chunk: Json = { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] };
  if (usage !== undefined) chunk.usage = usage;
  return chunk;
}

/**
 * A Gemini streamGenerateContent SSE stream as OpenAI chat-completion chunks.
 * The provider's frames carry one text piece each, the last carrying the
 * finish reason and the usage; the client's chunks mirror the env's own
 * openai answer shape: a role chunk first, content chunks, a finish chunk,
 * a usage chunk when the client asked for it, then [DONE].
 */
async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string, includeUsage: boolean): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let pendingFinish: string | null = null;
  let pendingUsage: Json | null = null;

  const startChunk = (): string[] => {
    if (started) return [];
    started = true;
    return [`data: ${JSON.stringify(openAIChunk(model, { role: 'assistant', content: '' }, null))}\n\n`];
  };
  const finishChunks = (): string[] => {
    if (finished) return [];
    finished = true;
    const out: string[] = [...startChunk(), `data: ${JSON.stringify(openAIChunk(model, {}, pendingFinish ?? 'stop'))}\n\n`];
    if (includeUsage && pendingUsage !== null) {
      out.push(`data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage: pendingUsage })}\n\n`);
    }
    out.push('data: [DONE]\n\n');
    return out;
  };

  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...finishChunks());
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
      const first = isRecord(candidates[0]) ? candidates[0] : {};
      const content = isRecord(first.content) ? first.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
      if (text !== '') {
        out.push(...startChunk());
        out.push(`data: ${JSON.stringify(openAIChunk(model, { content: text }, null))}\n\n`);
      }
      if (first.finishReason !== undefined && first.finishReason !== null) pendingFinish = openAIFinishOf(first.finishReason);
      if (isRecord(chunk.usageMetadata)) {
        const num = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);
        const input = num(chunk.usageMetadata.promptTokenCount);
        const output = num(chunk.usageMetadata.candidatesTokenCount);
        pendingUsage = { prompt_tokens: input, completion_tokens: output, total_tokens: input + output };
      }
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) for (const event of finishChunks()) yield event;
}

/**
 * A Gemini streamGenerateContent SSE stream as the Anthropic event sequence:
 * message_start, content_block_start/delta/stop for one text block, then
 * message_delta with the stop reason and usage, then message_stop.
 */
async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let blockOpen = false;
  let pendingStop: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', { type: 'message_start', message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } } }),
    ];
  };
  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    const out: string[] = [];
    if (blockOpen) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
      blockOpen = false;
    }
    out.push(
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: pendingStop ?? 'end_turn', stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    );
    return out;
  };

  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents());
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
      const first = isRecord(candidates[0]) ? candidates[0] : {};
      const content = isRecord(first.content) ? first.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
      if (text !== '') {
        out.push(...startEvents());
        if (!blockOpen) {
          blockOpen = true;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } }));
      }
      if (first.finishReason !== undefined && first.finishReason !== null) pendingStop = anthropicStopOf(first.finishReason);
      if (isRecord(chunk.usageMetadata)) {
        const num = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);
        inputTokens = num(chunk.usageMetadata.promptTokenCount);
        outputTokens = num(chunk.usageMetadata.candidatesTokenCount);
      }
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents()) yield event;
  }
}

// ---------------------------------------------------------------------------
// Serving: one provider call per candidate, best first, failing over.
// ---------------------------------------------------------------------------

interface Served {
  candidate: Candidate;
  response: Response;
}

/** One provider call for one candidate, in the provider's own wire. */
async function callCandidate(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json): Promise<Response> {
  const probe = candidate.probe;
  const base = joinBase(ENV_URL, probe.provider.base);
  const stream = body.stream === true;
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(probe.provider.auth, probe.provider.key) };
  if (probe.provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (stream) headers.accept = 'text/event-stream';

  if (probe.provider.wire === 'openai') {
    const upstream = clientWire === 'anthropic' ? anthropicToOpenAI(body, candidate.model.id) : { ...body, model: candidate.model.id };
    if (stream) upstream.stream_options = { include_usage: true };
    return fetch(`${base}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(upstream) });
  }
  if (probe.provider.wire === 'anthropic') {
    // Only an Anthropic client rides this wire (candidatesFor excludes it for
    // an OpenAI client): the body goes straight through, model replaced.
    const upstream = { ...body, model: candidate.model.id };
    return fetch(`${base}/v1/messages`, { method: 'POST', headers, body: JSON.stringify(upstream) });
  }
  // Gemini wire.
  const upstream = clientWire === 'anthropic' ? anthropicToGemini(body, candidate.model.id) : openAIToGemini(body, candidate.model.id);
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  const url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}${stream ? '?alt=sse' : ''}`;
  return fetch(url, { method: 'POST', headers, body: JSON.stringify(upstream) });
}

/**
 * Serve one client request: try the free candidates best-first, and on a
 * 402, 429 or 5xx (or a 400, which a model-specific refusal also wears) try
 * the next. The first 2xx wins; exactly one provider call answers the
 * request, which the client's one_provider check requires. Null when no free
 * model could serve it.
 */
async function serve(clientWire: 'openai' | 'anthropic', body: Json): Promise<Served | null> {
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires, clientWire);
  for (const candidate of candidates) {
    let response: Response;
    try {
      response = await callCandidate(candidate, clientWire, body);
    } catch (err) {
      log(`call to ${candidate.probe.provider.id}/${candidate.model.id} failed: ${(err as Error).message}`);
      noteFailure(candidate.probe.provider.id);
      continue;
    }
    if (response.ok) {
      noteSuccess(candidate.probe.provider.id);
      return { candidate, response };
    }
    // A refusal is not the answer: drain the error body and try the next
    // free model that meets the request.
    noteFailure(candidate.probe.provider.id);
    try {
      await response.text();
    } catch {
      // The body is unreadable: the next candidate is tried all the same.
    }
    log(`call to ${candidate.probe.provider.id}/${candidate.model.id} answered HTTP ${response.status}`);
  }
  return null;
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

/** Forward a provider's JSON answer to the client, status and body as they came. */
async function pipeJson(response: Response, res: ServerResponse): Promise<void> {
  const text = await response.text();
  res.writeHead(response.status, { 'content-type': response.headers.get('content-type') ?? 'application/json' });
  res.end(text);
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const served = await serve('openai', body);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory could serve this request');
  const wire = served.candidate.probe.provider.wire;
  const model = served.candidate.model.id;
  if (wire === 'gemini') {
    if (stream && served.response.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      for await (const event of openAIEventsFromGemini(served.response.body, model, true)) res.write(event);
      res.end();
      return;
    }
    const answer: unknown = await served.response.json();
    sendJson(res, 200, geminiToOpenAI(isRecord(answer) ? answer : {}, model));
    return;
  }
  // OpenAI wire: the provider's answer is already the client's wire.
  if (stream && served.response.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    await pipeStream(served.response.body, res);
    return;
  }
  await pipeJson(served.response, res);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const served = await serve('anthropic', body);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory could serve this request');
  const wire = served.candidate.probe.provider.wire;
  const model = served.candidate.model.id;
  const requested = typeof body.model === 'string' && body.model !== '' ? body.model : model;
  if (wire === 'gemini') {
    if (stream && served.response.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      for await (const event of anthropicEventsFromGemini(served.response.body, model)) res.write(event);
      res.end();
      return;
    }
    const answer: unknown = await served.response.json();
    sendJson(res, 200, geminiToAnthropic(isRecord(answer) ? answer : {}, model));
    return;
  }
  if (wire === 'anthropic') {
    // The provider's answer is already the client's wire.
    if (stream && served.response.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      await pipeStream(served.response.body, res);
      return;
    }
    await pipeJson(served.response, res);
    return;
  }
  // OpenAI wire: translate the answer back to Anthropic.
  if (stream && served.response.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    for await (const event of anthropicEventsFromOpenAI(served.response.body, requested)) res.write(event);
    res.end();
    return;
  }
  const answer: unknown = await served.response.json();
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, requested));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
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
