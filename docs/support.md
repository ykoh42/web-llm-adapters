# WebLLM support

Scope: WebLLM **0.2.86** and Strands SDK **1.21.0**. “Native” means use the engine
you own, also exposed as `model.engine`. It does not mean Strands gained a matching
API or that every model accepts the feature. All examples are type checked and built.

| Public feature group                                | Access                                                | Example / verification                                              |
| --------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------- |
| Chat, system prompt, history                        | Strands messages and `systemPrompt`                   | Chat example; unit tests and real Gemma streaming                   |
| Stream events, stop reasons, token usage            | Strands events                                        | Unit tests and browser tests                                        |
| Cancel and early consumer return                    | Cancel signal / `agent.cancel()`                      | Unit tests; browser stop and reuse check                            |
| Tools and results                                   | Strands `FunctionTool` and native WebLLM tools        | Real Hermes tool batch; mocked multi-tool Agent loop                |
| JSON and JSON schema                                | `request.response_format`                             | JSON example; schema and parsed output checks                       |
| EBNF grammar                                        | `request.response_format`                             | Grammar example                                                     |
| Structural tags                                     | `request.response_format`                             | Native type passed through; no real model check                     |
| Sampling, seed, penalties, stop strings, logit bias | `request`; common fields also use Strands config      | Mapping tests; JSON example uses seed and logprobs                  |
| Logprobs, top logprobs, latency breakdown           | `request` and `onChunk`                               | JSON example; raw chunk and metadata tests                          |
| Thinking flag                                       | `request.extra_body.enable_thinking`                  | Passed through; model specific; not tested on Qwen3                 |
| Image URLs / bytes                                  | Strands `ImageBlock`                                  | URL mapping tests; real Phi image bytes                             |
| Base64 WAV                                          | Strands `AudioBlock` with WAV bytes                   | Mapping tests; real Gemma with synthetic silence                    |
| Raw PCM / other browser-decodable audio             | Native `chat.completions.create`                      | PCM example; Strands has no raw PCM block                           |
| Text completions and multiple choices               | Native `completions.create`                           | Two-choice completion example                                       |
| Embeddings                                          | Native `embeddings.create`                            | Dedicated Snowflake example                                         |
| Persist, list, recover, continue, delete generation | Native resumable API; request enables persistence     | Save + resume example; orderly interruption only                    |
| Main thread, Web Worker, Service Worker             | Inject an `MLCEngineInterface`                        | Real Gemma in all three; stop/reuse in both workers                 |
| Chrome extension workers                            | Inject WebLLM's extension engine                      | Native interface; no extension example or browser check             |
| Loading progress, unload, model change              | Native engine lifecycle                               | Engine selector and load/unload buttons                             |
| Cache API, IndexedDB, OPFS                          | Native `AppConfig` and cache helpers                  | Real load, cache check, and deletion in each backend                |
| Cross-origin cache                                  | Native `AppConfig` plus the separate extension        | Not exercised; requires extension setup                             |
| Custom models, manifests, integrity hashes          | Native `AppConfig.model_list`                         | Native configuration; no custom-artifact verification here          |
| Multiple loaded models                              | Native `reload([ids])`; adapter `modelId` selects one | Model selection mapping test; no simultaneous multi-model GPU check |
| Custom logit processors                             | Native `MLCEngineConfig.logitProcessorRegistry`       | Main-thread engine only; no processor example                       |
| GPU info, logging, stats, current message, reset    | Native engine methods                                 | GPU info shown on load; usage preferred over legacy stats           |
| Token-level forwarding and sampling                 | Native `forwardTokensAndSample`                       | Low-level API; not a Strands operation or tested here               |

## Boundaries

The adapter owns one native generation at a time, not the engine or its models.
It closes the native iterator on abort even when Strands leaves its outer iterator
suspended. A new adapter call waits for that cancelled call's cleanup. Concurrent
active calls are rejected. Native calls and `reload()` must not overlap an adapter
call; the adapter cannot guard work made directly through the engine.

Strands controls messages, tools, streaming, and one response per request. Native
non-streaming calls, `n > 1`, text completion, embeddings, and resume keep their own
APIs. `agent.invoke()` still uses a streaming model request internally. A persisted
WebLLM generation does not persist a Strands Agent, its tools, or its history.

Only Hermes models in WebLLM's `functionCallingModelIds` accept native tools.
Their tool mode adds a system prompt and a JSON grammar, so custom system prompts
and response formats are rejected upstream. `toolChoice.any` has no native equivalent.
The adapter does not invent a tool parser for Gemma. Tool JSON is buffered until
WebLLM supplies complete parsed calls; truncated calls are never executed.

Image and audio inputs require a model that accepts them. User text blocks are
joined because WebLLM accepts one text part per message. Tool results are placed
before injected user text. Strands cache-point hints do not configure WebLLM's KV
cache. Unsupported content raises an error: S3 media, non-WAV Strands audio,
video, documents, guardrails, reasoning blocks, and media in tool results.

## Verification

Verified on 2026-10-16 with Chrome 155 on macOS and WebGPU:

- **37 CPU tests**: native request validation, real Strands Agent loops with a
  mocked engine, delayed chunks, cancellation, cleanup, and example output.
- **15 browser cases**: main-thread streaming; Web Worker and Service Worker stop
  and reuse; JSON schema with logprobs; EBNF; save and resume; two completions;
  WAV and PCM; 384-dimensional embeddings; three cache backends; one Hermes tool
  batch; and a red image described by Phi vision.
- Gemma 4 E2B, Hermes 3 Llama 3.1 8B, and Phi 3.5 vision used their FP16 variants.
  Embeddings used `snowflake-arctic-embed-s-q0f32-MLC-b4`.

`npm run check` runs CPU checks in CI. The mocked engine tests use real Strands
interfaces; they validate mapping, ordering, and lifetime behavior, not GPU inference.
`npm run test:browser` runs the actual page and models separately on Chrome/WebGPU.
Synthetic silence in audio tests checks decoding and inference, not transcription
quality. Native API access alone is not a 100% support or verification claim.

Sources: [installed API source](https://github.com/mlc-ai/web-llm/tree/v0.2.86/src),
[release notes](https://github.com/mlc-ai/web-llm/releases/tag/v0.2.86),
[upstream examples](https://github.com/mlc-ai/web-llm/tree/v0.2.86/examples).
