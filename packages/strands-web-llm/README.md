# strands-web-llm

A community WebLLM provider for Strands SDK and Harness. Run inference in a
WebGPU browser; use the same engine from Strands and WebLLM's native APIs.

See the [browser examples](https://github.com/ykoh42/web-llm-adapters/blob/main/examples/strands-web-llm/browser/README.md)
or [Node.js Harness example](https://github.com/ykoh42/web-llm-adapters/blob/main/examples/strands-web-llm/harness/README.md).

## Models

These exact WebLLM **0.2.86** IDs were tested in a real browser Harness run on
Apple M4 Pro with Chrome. “Pass” means tool call, result, and final answer. IDs match WebLLM's official
[`model_list`](https://github.com/mlc-ai/web-llm/blob/v0.2.86/src/config.ts).

| Exact WebLLM `model_id`             | Tool calls | Separate thinking                |
| ----------------------------------- | ---------- | -------------------------------- |
| `Qwen2.5-0.5B-Instruct-q4f16_1-MLC` | Pass       | Not tested                       |
| `Qwen2.5-1.5B-Instruct-q4f16_1-MLC` | Pass       | Not tested                       |
| `Qwen2.5-3B-Instruct-q4f16_1-MLC`   | Pass       | Not tested                       |
| `Qwen2.5-7B-Instruct-q4f16_1-MLC`   | Pass       | Not tested                       |
| `Qwen3-0.6B-q4f16_1-MLC`            | Pass       | Separate Strands reasoning block |
| `Qwen3-4B-q4f16_1-MLC`              | Pass       | Separate Strands reasoning block |
| `Qwen3-8B-q4f16_1-MLC`              | Pass       | Separate Strands reasoning block |
| `Qwen3.5-0.8B-q4f16_1-MLC`          | Pass       | Separate Strands reasoning block |
| `Qwen3.5-2B-q4f16_1-MLC`            | Pass       | Separate Strands reasoning block |
| `Qwen3.5-4B-q4f16_1-MLC`            | Pass       | Separate Strands reasoning block |
| `Qwen3.5-9B-q4f16_1-MLC`            | Pass       | Separate Strands reasoning block |
| `gemma-4-E2B-it-q4f16_1-MLC`        | Pass       | Separate block, on or off        |

Qwen3 1.7B and Phi-4 mini are excluded because they did not complete reliable
tool runs. Other catalog IDs are unverified for a complete tool turn. The
Qwen adapters follow the official [Qwen2.5](https://qwen.readthedocs.io/en/v2.5/framework/function_call.html),
[Qwen3](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/concepts.md),
and [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/chat_template.jinja) tool formats.
The adapter sends model-specific tool schemas and call formats in the system
message, then parses the generated calls for Harness. WebLLM's five native-tool
Hermes IDs made tool calls but did not finish the tested turn with a final answer.

Qwen tool calls use model-generated text, so the adapter cannot force a
model to follow the format or a requested tool choice. “Pass” means that exact
Harness run completed; it is not a guarantee for every prompt. Calls are passed
through as generated, including repeated calls.
A prior Qwen2.5 3B run returned an invalid tool name; subsequent runs passed.

### Other verified model types

| Exact WebLLM `model_id`                 | Strands API   | Verified result                       | Limit                |
| --------------------------------------- | ------------- | ------------------------------------- | -------------------- |
| `Phi-3.5-vision-instruct-q4f16_1-MLC`   | `ImageBlock`  | Harness reads a generated image       | See model list       |
| `snowflake-arctic-embed-s-q0f32-MLC-b4` | `MemoryStore` | Semantic search and Harness injection | 384 dims; 512 tokens |

WebLLM embeddings are not a Strands chat `Model` or content block.
`WebLLMEmbeddingSearchStrategy` implements Strands [`SearchStrategy`](https://strandsagents.com/docs/user-guide/sdk/storage/search-strategies/)
for storage and memory search. The [Harness example](../../examples/strands-web-llm/harness/src/memory.ts)
uses `engine.embeddings` through `withEngine()`. For Snowflake Arctic Embed S, prefix search
queries with `Represent this sentence for searching relevant passages: `; stored text stays unprefixed.
The strategy does not split long text. Its vector cache rebuilds from saved text after restart,
and each search reads every stored entry. Use it for small memory stores, not large knowledge bases.
Other vision and embedding IDs are unverified.

The Qwen parser accepts complete calls when a model omits the closing tag or adds
one extra closing brace. It rejects other malformed calls and never changes
parsed tool names or arguments. WebLLM loads each model's Wasm through its
`model_lib` URL. The bundled Wasm fixes Gemma 4 E2B specifically.

## Install

```sh
npm install strands-web-llm @mlc-ai/web-llm @strands-agents/sdk
```

## Browser

This example uses Qwen3 0.6B. For Gemma 4 E2B, use the
[required runtime setup](runtime/README.md#use).

```ts
import { CreateMLCEngine } from '@mlc-ai/web-llm'
import { Agent } from '@strands-agents/sdk'
import { WebLLMModel } from 'strands-web-llm'

const modelId = 'Qwen3-0.6B-q4f16_1-MLC'
const engine = await CreateMLCEngine(modelId)
const model = new WebLLMModel({ engine, modelId })
const agent = new Agent({ model })

try {
  const result = await agent.invoke('Say hello.')
  console.log(result.lastMessage.content)
} finally {
  await engine.unload()
}
```

The application owns loading and unloading. WebLLM's Worker factories work too.

## Node.js Harness

On your browser page, expose the loaded engine:

```ts
import { createBrowserBridge } from 'strands-web-llm'

window.webllmBridge = createBrowserBridge(engine)
```

In Node.js, pass its Playwright page to Harness:

```ts
import { createHarness } from '@strands-agents/harness'
import { BrowserModel } from 'strands-web-llm'

const model = new BrowserModel(page, {
  modelId, // The model loaded in the browser.
  contextWindowLimit: 16384, // Match the loaded engine.
  maxTokens: 4096,
})
const agent = await createHarness({ model })
try {
  console.log((await agent.invoke('Say hello.')).lastMessage.content)
} finally {
  await agent.shutdown()
}
```

Tools run in Node.js; inference runs in the browser. Before unloading or closing
the page, call `await page.evaluate(() => window.webllmBridge.close())`.
Harness tools use your host permissions unless you configure a sandbox.
Use a page controlled by your application. The browser app uses TypeScript
`moduleResolution: "Bundler"`; Node.js callers can use `module: "NodeNext"`.

## Native WebLLM APIs

Use `withEngine()` in tools, hooks, or application code to share the engine safely:

```ts
import { withEngine } from 'strands-web-llm'

const result = await withEngine(engine, (engine) =>
  engine.embeddings.create({ model: embeddingModelId, input: 'hello' }),
)
```

For Node.js, run the callback on the browser page:

```ts
const stats = await page.evaluate(() =>
  window.webllmBridge.withEngine((engine) => engine.runtimeStatsText()),
)
```

The callback receives the original engine. Await all work and consume streams
inside it. Return serializable results to Node.js. Calls sharing an engine wait
in order; do not invoke an Agent or another scope on that engine inside the callback.
Stop callers before replacing their loaded models. Direct engine calls bypass the queue.

## Options

Both models accept native generation settings in `request` and original output
chunks through `onChunk`. SDK `maxTokens`, `temperature`, and `topP` take precedence.
`updateConfig()` merges settings; `modelId` selects an already loaded model.

Use `agent.stream()` for events and `agent.cancel()` to stop generation. Gemma,
Qwen3, and Qwen3.5 return separate Strands reasoning blocks when thinking is on.
Qwen keeps constrained JSON output unchanged. Render only `textBlock` values to hide reasoning.
`pcmAudio(samples, sampleRate)` converts mono float32 PCM to a lossless WAV `AudioBlock`.

## Support

Targets: WebLLM **0.2.86**, Strands SDK **1.21.0**, Harness **0.3.0**.
“Native” below means WebLLM's original API inside `withEngine()`.

| Feature                                                              | Connection                                                        | Limits                                                           |
| -------------------------------------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------- |
| Text, system prompts, history                                        | Strands messages and `systemPrompt`                               | Unsupported block types raise errors                             |
| Model selection                                                      | `modelId`                                                         | Model must already be loaded                                     |
| Token limit, sampling, penalties, seed, stop, logit bias             | `request`; SDK config aliases                                     | SDK settings take precedence                                     |
| Logprobs and original chunks                                         | `request`, `onChunk`                                              | Available in both providers                                      |
| Usage, latency, timing                                               | Strands metadata and `onChunk`                                    | Native extra fields retained                                     |
| Plain text, JSON schema, grammar, structural tags                    | `request.response_format`                                         | Gemma protocol mode accepts plain text only                      |
| Tool declarations, results, auto/named/none choice                   | Strands tools and `toolChoice`; `request.tool_choice: 'none'`     | `any` unsupported; model must support tools                      |
| Thinking                                                             | `request.extra_body.enable_thinking`                              | Depends on the exact model ID; see the model table above         |
| Images: URL, data URL, bytes                                         | SDK `ImageBlock`                                                  | Only the verified vision ID above is tested                      |
| WAV and float32 PCM                                                  | SDK `AudioBlock`, `pcmAudio()`                                    | Needs an audio model; WAV bytes only at the native chat boundary |
| `ignore_eos`                                                         | `request.ignore_eos`                                              | Unavailable in Gemma protocol mode                               |
| Streaming and final response                                         | `agent.stream()`, `agent.invoke()`                                | One assistant response per model call                            |
| Multiple choices, non-streaming backend                              | Native chat/completion                                            | WebLLM rejects multiple streamed choices                         |
| Raw prompt and echo                                                  | Native completions                                                | Separate from Strands message turns                              |
| Embeddings                                                           | Strands `MemoryStore` / `SearchStrategy` or tool + `withEngine()` | Not a chat model; see the verified ID above                      |
| Message `name`                                                       | Native chat                                                       | SDK messages have no name field                                  |
| Generation persistence                                               | Native chat `request.extra_body.resumable`; native recovery APIs  | Separate from Strands sessions; no raw-completion recovery       |
| Reload/unload, reset, progress, logging, message, stats, GPU queries | Native engine                                                     | Caller owns lifecycle                                            |
| Token-level sampling and logit processors                            | Native engine and setup                                           | Keep processor setup in WebLLM                                   |
| Cache backends, inspection, deletion, integrity, custom models       | Native WebLLM setup/cache APIs                                    | Backend and model restrictions apply                             |
| Main thread, Web Worker, Service Worker, extension worker            | Any `MLCEngineInterface`                                          | Set up workers with WebLLM's factories and handlers              |
| Context counting and overflow                                        | Strands `countTokens()` and recovery error                        | Counts are estimates; set the loaded context window size         |

## Limits

Features depend on the loaded model and WebLLM. Invalid model output and token-limit
errors reach the caller. Qwen accepts complete JSON calls with a missing closing
tag or one extra closing brace; other malformed calls fail. The adapter never
changes parsed tool names or arguments or supplies answers.
Qwen prompt-based tool calls cannot share a request with a structured `response_format`.
Structured output is passed through unchanged, including text that looks like protocol tags.
For Gemma 4 E2B on WebLLM 0.2.86, see the [required cache fix and protocol limits](runtime/README.md).

S3 media, video, documents, guardrails, and media in tool results are unsupported.
Cache hints do not configure WebLLM's KV cache. Signed/redacted reasoning is unsupported.
Strands sessions save Agent history; WebLLM checkpoints save native generation.
Recovering an interrupted external tool effect requires application-level handling.

See the [runnable examples](https://github.com/ykoh42/web-llm-adapters/blob/main/examples/strands-web-llm/browser/README.md)
and [Harness flows](https://github.com/ykoh42/web-llm-adapters/blob/main/examples/strands-web-llm/harness/README.md)
for complete setups and tests. The [native engine interface](https://github.com/mlc-ai/web-llm/blob/v0.2.86/src/types.ts)
and [Strands provider contract](https://strandsagents.com/docs/user-guide/sdk/model-providers/custom_model_provider/)
define each library's API boundary.

## License

[MIT](LICENSE). The bundled runtime retains its [upstream licenses and notices](runtime/NOTICE.txt).
