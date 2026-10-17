# strands-web-llm

A community WebLLM provider for Strands SDK and Harness. Run inference in a
WebGPU browser; use the same engine from Strands and WebLLM's native APIs.

Use a model from WebLLM’s catalog; feature support depends on that model.
See the [browser examples](https://github.com/ykoh42/web-llm-adapters/blob/main/examples/strands-web-llm/browser/README.md)
or [Node.js Harness example](https://github.com/ykoh42/web-llm-adapters/blob/main/examples/strands-web-llm/harness/README.md).

## Install

```sh
npm install strands-web-llm @mlc-ai/web-llm @strands-agents/sdk
```

## Browser

Choose a chat `modelId` from WebLLM's catalog:

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

Use `agent.stream()` for events and `agent.cancel()` to stop generation. Set
`request.extra_body.enable_thinking` for Gemma reasoning; thinking and answer text
remain separate. To print only the answer, select `textBlock` values.
`pcmAudio(samples, sampleRate)` converts mono float32 PCM to a lossless WAV `AudioBlock`.

## Support

Targets: WebLLM **0.2.86**, Strands SDK **1.21.0**, Harness **0.3.0**.
“Native” below means WebLLM's original API inside `withEngine()`.

| Feature                                                              | Connection                                                       | Limits                                                               |
| -------------------------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------- |
| Text, system prompts, history                                        | Strands messages and `systemPrompt`                              | Unsupported block types raise errors                                 |
| Model selection                                                      | `modelId`                                                        | Model must already be loaded                                         |
| Token limit, sampling, penalties, seed, stop, logit bias             | `request`; SDK config aliases                                    | SDK settings take precedence                                         |
| Logprobs and original chunks                                         | `request`, `onChunk`                                             | Available in both providers                                          |
| Usage, latency, timing                                               | Strands metadata and `onChunk`                                   | Native extra fields retained                                         |
| Plain text, JSON schema, grammar, structural tags                    | `request.response_format`                                        | Gemma protocol mode accepts plain text only                          |
| Tool declarations, results, auto/named/none choice                   | Strands tools and `toolChoice`; `request.tool_choice: 'none'`    | `any` unsupported; model must support tools                          |
| Thinking                                                             | `request.extra_body.enable_thinking`                             | Gemma uses Strands reasoning blocks; other models keep native output |
| Images: URL, data URL, bytes                                         | SDK `ImageBlock`                                                 | Needs a vision model                                                 |
| WAV and float32 PCM                                                  | SDK `AudioBlock`, `pcmAudio()`                                   | Needs an audio model; WAV bytes only at the native chat boundary     |
| `ignore_eos`                                                         | `request.ignore_eos`                                             | Unavailable in Gemma protocol mode                                   |
| Streaming and final response                                         | `agent.stream()`, `agent.invoke()`                               | One assistant response per model call                                |
| Multiple choices, non-streaming backend                              | Native chat/completion                                           | WebLLM rejects multiple streamed choices                             |
| Raw prompt and echo                                                  | Native completions                                               | Separate from Strands message turns                                  |
| Embeddings                                                           | Native embeddings                                                | Use a loaded embedding model                                         |
| Message `name`                                                       | Native chat                                                      | SDK messages have no name field                                      |
| Generation persistence                                               | Native chat `request.extra_body.resumable`; native recovery APIs | Separate from Strands sessions; no raw-completion recovery           |
| Reload/unload, reset, progress, logging, message, stats, GPU queries | Native engine                                                    | Caller owns lifecycle                                                |
| Token-level sampling and logit processors                            | Native engine and setup                                          | Keep processor setup in WebLLM                                       |
| Cache backends, inspection, deletion, integrity, custom models       | Native WebLLM setup/cache APIs                                   | Backend and model restrictions apply                                 |
| Main thread, Web Worker, Service Worker, extension worker            | Any `MLCEngineInterface`                                         | Set up workers with WebLLM's factories and handlers                  |
| Context counting and overflow                                        | Strands `countTokens()` and recovery error                       | Counts are estimates; set the loaded context window size             |

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
