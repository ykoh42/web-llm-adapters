# strands-web-llm

A community WebLLM provider for Strands SDK and Harness. Run inference in a
WebGPU browser; use the same engine from Strands and WebLLM's native APIs.

See the [browser examples](https://github.com/ykoh42/web-llm-adapters/blob/main/examples/strands-web-llm/browser/README.md)
or [Node.js Harness example](https://github.com/ykoh42/web-llm-adapters/blob/main/examples/strands-web-llm/harness/README.md).

## Models

The IDs below match entries in WebLLM **0.2.86**'s official
[`prebuiltAppConfig.model_list`](https://github.com/mlc-ai/web-llm/blob/v0.2.86/src/config.ts).
Each row names one exact model and quantization. Results are from real browser
runs on an Apple M4 Pro with Chrome.

| Exact WebLLM `model_id`                 | Text / media              | Strands tool calls                                                                | Thinking                                                                                     |
| --------------------------------------- | ------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `Llama-3.2-1B-Instruct-q4f16_1-MLC`     | Text works                | Rejected by WebLLM 0.2.86 for this ID                                             | No separate thought output observed                                                          |
| `Qwen3-0.6B-q4f16_1-MLC`                | Text works                | Rejected by WebLLM 0.2.86 for this ID                                             | Works; native `<think>` text stays in the answer, not a separate Strands reasoning block     |
| `gemma-4-E2B-it-q4f16_1-MLC`            | Text works                | Works through the adapter, including tool result and final answer                 | Works; thinking is a separate Strands reasoning block                                        |
| `Phi-3.5-vision-instruct-q4f16_1-MLC`   | Text and image input work | Rejected by WebLLM 0.2.86 for this ID                                             | No separate thought output; `false` emitted empty `<think>` blocks until the token limit     |
| `Hermes-3-Llama-3.1-8B-q4f16_1-MLC`     | Text works                | One call runs; then this ID repeated calls for three turns without a final answer | No reliable switch; `false` exposed raw `<think>` text, while `true` returned a plain answer |
| `snowflake-arctic-embed-s-q0f32-MLC-b4` | 384-value embedding works | Not a chat model                                                                  | Not a chat model                                                                             |

The three “Rejected by WebLLM” results are its own model-ID checks. The Hermes
repeat was reproduced through WebLLM's native API too. Its example ends after one
tool batch because this model did not produce a final answer in the tested loop.
Other sizes and quantizations are separate IDs and are not covered by these results.
Custom model IDs use WebLLM's `appConfig.model_list` and need their own validation.

WebLLM loads each model's Wasm through its `model_lib` URL. The bundled Wasm fixes
Gemma 4 E2B specifically.

## Install

```sh
npm install strands-web-llm @mlc-ai/web-llm @strands-agents/sdk
```

## Browser

Choose a chat `modelId` from the [model list](#models):

```ts
import { CreateMLCEngine } from '@mlc-ai/web-llm'
import { Agent } from '@strands-agents/sdk'
import { WebLLMModel } from 'strands-web-llm'

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

Use `agent.stream()` for events and `agent.cancel()` to stop generation.
Thinking depends on the model: Gemma returns separate Strands reasoning blocks;
Qwen returns WebLLM's native `<think>` text with the answer.
To omit Gemma reasoning, render only `textBlock` values. Qwen's thinking stays in
the same text as its answer.
`pcmAudio(samples, sampleRate)` converts mono float32 PCM to a lossless WAV `AudioBlock`.

## Support

Targets: WebLLM **0.2.86**, Strands SDK **1.21.0**, Harness **0.3.0**.
“Native” below means WebLLM's original API inside `withEngine()`.

| Feature                                                              | Connection                                                       | Limits                                                           |
| -------------------------------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------- |
| Text, system prompts, history                                        | Strands messages and `systemPrompt`                              | Unsupported block types raise errors                             |
| Model selection                                                      | `modelId`                                                        | Model must already be loaded                                     |
| Token limit, sampling, penalties, seed, stop, logit bias             | `request`; SDK config aliases                                    | SDK settings take precedence                                     |
| Logprobs and original chunks                                         | `request`, `onChunk`                                             | Available in both providers                                      |
| Usage, latency, timing                                               | Strands metadata and `onChunk`                                   | Native extra fields retained                                     |
| Plain text, JSON schema, grammar, structural tags                    | `request.response_format`                                        | Gemma protocol mode accepts plain text only                      |
| Tool declarations, results, auto/named/none choice                   | Strands tools and `toolChoice`; `request.tool_choice: 'none'`    | `any` unsupported; model must support tools                      |
| Thinking                                                             | `request.extra_body.enable_thinking`                             | Depends on the exact model ID; see the model table above         |
| Images: URL, data URL, bytes                                         | SDK `ImageBlock`                                                 | Needs a vision model                                             |
| WAV and float32 PCM                                                  | SDK `AudioBlock`, `pcmAudio()`                                   | Needs an audio model; WAV bytes only at the native chat boundary |
| `ignore_eos`                                                         | `request.ignore_eos`                                             | Unavailable in Gemma protocol mode                               |
| Streaming and final response                                         | `agent.stream()`, `agent.invoke()`                               | One assistant response per model call                            |
| Multiple choices, non-streaming backend                              | Native chat/completion                                           | WebLLM rejects multiple streamed choices                         |
| Raw prompt and echo                                                  | Native completions                                               | Separate from Strands message turns                              |
| Embeddings                                                           | Native embeddings                                                | Use a loaded embedding model                                     |
| Message `name`                                                       | Native chat                                                      | SDK messages have no name field                                  |
| Generation persistence                                               | Native chat `request.extra_body.resumable`; native recovery APIs | Separate from Strands sessions; no raw-completion recovery       |
| Reload/unload, reset, progress, logging, message, stats, GPU queries | Native engine                                                    | Caller owns lifecycle                                            |
| Token-level sampling and logit processors                            | Native engine and setup                                          | Keep processor setup in WebLLM                                   |
| Cache backends, inspection, deletion, integrity, custom models       | Native WebLLM setup/cache APIs                                   | Backend and model restrictions apply                             |
| Main thread, Web Worker, Service Worker, extension worker            | Any `MLCEngineInterface`                                         | Set up workers with WebLLM's factories and handlers              |
| Context counting and overflow                                        | Strands `countTokens()` and recovery error                       | Counts are estimates; set the loaded context window size         |

## Limits

Features depend on the loaded model and WebLLM. Invalid model output and token-limit
errors reach the caller. The adapter does not repair tool arguments or supply answers.
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
