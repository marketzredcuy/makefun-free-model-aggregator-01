# free-model-aggregator-01 — a quality-ranked free-model router

One server, two client wires (OpenAI Chat Completions, Anthropic Messages),
ten simulated free-tier providers on three upstream wires (openai, anthropic,
gemini). It reads the brief, probes every provider's catalog route, builds
one catalog, registers with the env, and answers the client's 20 requests,
one at a time, each with the best free model that meets the request's
requirements and still has budget — failing over to the next-best model when
a provider refuses.

## Run

```sh
node main.ts            # serve: reads $MAKEFUN_PROMPT_FILE, registers at $MAKEFUN_ENV_URL
node main.ts --list     # print the catalog JSON and exit 0
```

Node 22, standard library only. No dependencies, no model calls of its own.

## How it routes (quality is 60% of the score)

Every free model that meets a request's requirements is ranked by the
quality table's `rating_milli`, best first, and tried in that order. A
provider that answers 402, 429 or 5xx is skipped for that request and the
next candidate is tried, so a refusal never kills a request while another
free model can serve it. A provider is never blacklisted across requests:
a refusal may be a scheduled failure, not a spent budget, so the next
request tries the best candidate again — unless the provider has refused
three times in a row, which is a spent budget or an outage (a trial's
failure schedules are one or two calls), in which case it is not tried
again. That keeps the provider-call count low, which is the tie-break.

## The wires

- **openai** — the client's body, with the routed model's id, straight
  through. Streaming sets `stream_options.include_usage` so the usage
  chunk the client checks for is there.
- **anthropic** — for an Anthropic client, the body straight through
  (same wire). An OpenAI client is never routed here: the wire cannot
  carry `response_format`, so the provider's echo would lose the
  request's JSON mode. Cloudflare lists the same 1470 model on the
  openai wire, so nothing is lost.
- **gemini** — the request is rebuilt as `generateContent` (or
  `streamGenerateContent`): `systemInstruction`, `contents` with text
  and `inlineData` images, `functionDeclarations`, `generationConfig`
  carrying `maxOutputTokens`, `stopSequences` and `responseMimeType`.
  The answer is rebuilt as the client's wire. Gemini is the only wire
  with a free model that has both tools and vision, so the three vision
  requests ride it.

Every translation preserves the fields the provider echoes back —
system text, user text, tool names, max tokens, stop sequences, image,
JSON mode — field for field, so the client's echo checks pass. Stop
reasons map `STOP`/`end_turn` to `stop`, `MAX_TOKENS` to `length` /
`max_tokens`, and usage counts are carried as the provider reported
them, in JSON and in streaming.

## Determinism

Nothing reads the wall clock or unseeded randomness, so a live pass and
its replay make the same provider calls and leave the same state. The
only dates any reply carries are the fixed timestamps in `lib/wire.ts`.

## Practice scores (grade-local, --driver process)

| trial | score | quality | served | wire | failover | catalog | calls |
|------:|------:|--------:|-------:|-----:|---------:|--------:|------:|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 27 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 31 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 34 |

Quality 99 is the ceiling on these instances: the best non-vision model
(OpenRouter's 1475) has a per-trial budget of 10–12 calls for 17
non-vision requests, so at least five requests must take the 1470 model.
Replay matched byte for byte on all three trials.

## Files

- `main.ts` — the server: brief, catalog, routing, failover, Gemini
  upstream wire, register and poll. The entry's own code.
- `lib/providers.ts`, `lib/wire.ts` — vendored from the competition's
  starter kit, used as-is: the brief parser, the catalog probe and the
  OpenAI <-> Anthropic wire translation.
- `LICENSE` — MIT.
