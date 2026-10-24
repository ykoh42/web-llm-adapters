# strands-web-llm

A community WebLLM provider for Strands SDK and Harness. Run inference in a
WebGPU browser while tools run in your application.

## Install

```sh
npm install strands-web-llm
```

- Yarn: also install compatible versions of `@mlc-ai/web-llm`, `@strands-agents/sdk`, and the SDK peers listed by `yarn explain peer-requirements`.
- pnpm: if dependency scripts are blocked, run [`pnpm approve-builds`](https://pnpm.io/cli/approve-builds) to allow or deny them, then rerun `pnpm install`.

## Browser

Create a worker that runs WebLLM:

```ts
// worker.ts
import { WebWorkerMLCEngineHandler } from '@mlc-ai/web-llm'

const handler = new WebWorkerMLCEngineHandler()
self.onmessage = (message: MessageEvent) => handler.onmessage(message)
```

Load a model in the browser, then pass its engine to Strands:

```ts
import { CreateWebWorkerMLCEngine } from '@mlc-ai/web-llm'
import { Agent } from '@strands-agents/sdk'
import { WebLLMModel } from 'strands-web-llm'

const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
const contextWindowSize = 4096
const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
try {
  const engine = await CreateWebWorkerMLCEngine(worker, modelId, undefined, {
    context_window_size: contextWindowSize,
  })
  try {
    const model = new WebLLMModel({ engine, modelId, contextWindowLimit: contextWindowSize })
    const agent = new Agent({ model })
    console.log((await agent.invoke('Say hello.')).lastMessage.content)
  } finally {
    await engine.unload()
  }
} finally {
  worker.terminate()
}
```

You own model loading and unloading. The examples use Worker and Main thread engines; the adapter accepts any WebLLM `MLCEngineInterface`.
The first load fetches WebLLM's runtime and model files; the browser caches them for later use.
Inference runs locally.
See the [runnable examples](https://github.com/ykoh42/web-llm-adapters/blob/main/examples/strands-web-llm/README.md).

## Node.js Harness

Install Harness for this example:

```sh
npm install @strands-agents/harness@0.3.1
```

On a browser page controlled by your application, load the model with
`context_window_size: 16384` and expose its engine:

```ts
import { createBrowserBridge } from 'strands-web-llm'
window.webllmBridge = createBrowserBridge(engine)
```

Pass its Playwright page to Harness in Node.js:

```ts
import { createHarness } from '@strands-agents/harness'
import { BrowserModel } from 'strands-web-llm'

const model = new BrowserModel(page, {
  modelId: 'Qwen3.5-4B-q4f16_1-MLC',
  contextWindowLimit: 16384, // Match the loaded engine.
  maxTokens: 4096,
})
// Qwen3.5 is text-only, so keep media out of Harness's read tool.
const agent = await createHarness({ model, builtinTools: { read: { media: false } } })
try {
  console.log((await agent.invoke('Say hello.')).lastMessage.content)
} finally {
  await agent.shutdown()
  await page.evaluate(() => window.webllmBridge.close())
}
```

Tools use your host permissions unless you configure a Harness sandbox.
The [Harness example](https://github.com/ykoh42/web-llm-adapters/blob/main/examples/strands-web-llm/harness/README.md) includes browser startup and shutdown.

## Models and memory

See the [verified models](https://github.com/ykoh42/web-llm-adapters#models) for exact IDs and features.
For Qwen3.5, declare object/array types to preserve XML in JSON. History cannot preserve `>` in top-level argument names or `<parameter=` / `</parameter>` in raw strings.
Install WebLLM `0.2.79` or later in the `0.2.x` series. Gemma 4 E2B was verified with WebLLM `0.2.86` and needs the bundled [runtime setup and protocol limits](https://github.com/ykoh42/web-llm-adapters/blob/main/shared/runtime/README.md). Audio works in ordinary chat, without tools or thinking.

`WebLLMEmbeddingSearchStrategy` connects WebLLM to Strands memory; see the
[memory example](https://github.com/ykoh42/web-llm-adapters/blob/main/examples/strands-web-llm/harness/src/memory.ts).
This strategy embeds UTF-8 text only. Entries with invalid UTF-8 stay in storage but are excluded from semantic search.
These embedding models have a 512-token input limit. Prefix queries with
`Represent this sentence for searching relevant passages: `; leave stored text unprefixed.
Search returns positive cosine matches by default; set `minScore` to include lower scores.
`includeData: false` reduces memory use; FileMemoryStore rereads the returned matches.
Searches scan all entries, and vectors rebuild after restart. Keep each stored record within 512 tokens;
FileMemoryStore appends facts that share a heading.

## Support

Tested with WebLLM **0.2.79** and **0.2.86**, and Strands SDK **1.18.0** and **1.22.0**. Model availability depends on the installed WebLLM version. Harness **0.3.1** is used by the integration example.
Features depend on the exact model.

For Gemma 4, `request.extra_body.preserve_thinking: true` also keeps earlier tool-call reasoning; the default keeps only the current user turn.

Set `thinkingPrefilled: true` at construction if your loaded Qwen template already opens `<think>`. This only controls output parsing; it does not change the engine or prompt.

| Feature                               | Use                                                                                      | Limit                                                          |
| ------------------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Chat, system prompts, history         | Strands messages and `systemPrompt`                                                      | Unsupported blocks raise errors                                |
| Streaming and cancellation            | `agent.stream()`, `agent.cancel()`                                                       | One assistant response per call                                |
| Tools and results                     | Strands tools and `toolChoice`                                                           | Qwen supports `any`; Gemma/native tool models require one tool |
| Disable tools                         | `request.tool_choice: 'none'`                                                            | Keeps the Agent tool registry                                  |
| Thinking                              | `request.extra_body.enable_thinking`                                                     | Render `textBlock` values to show only the answer              |
| Sampling and token limits             | `request`; `maxTokens`, `temperature`, `topP`                                            | SDK aliases take precedence                                    |
| JSON schema, grammar, structural tags | `request.response_format`                                                                | Cannot combine with Qwen tools or Gemma protocol mode          |
| Images and WAV audio                  | `ImageBlock`, `AudioBlock`, `pcmAudio()`                                                 | Model-dependent; PCM is converted to WAV                       |
| Usage and timing                      | Strands metadata                                                                         | Token counts and latency                                       |
| Log probabilities                     | `request.logprobs: true`; stop-event `additionalModelResponseFields.logprobs`; `onChunk` | Native per-token values retained                               |
| Embeddings                            | `WebLLMEmbeddingSearchStrategy` or `withEngine()`                                        | Separate embedding model required                              |
| Context recovery                      | `countTokens()` and Strands overflow error                                               | Counts are estimates; set the loaded context window size       |
| Other native APIs                     | `withEngine()`                                                                           | Caller owns setup and lifecycle                                |

Signed/redacted reasoning, S3 media, video, documents, guardrails, and non-image tool-result media
are unsupported. Tool-result images follow the adjacent tool replies; no listed model is verified
for both tools and vision. Invalid or incomplete tool output reaches the caller; the adapter does not
change tool names or arguments or supply answers. Validate model-selected arguments: Strands runtime
validation requires a [Zod `inputSchema`](https://strandsagents.com/docs/user-guide/sdk/tools/custom-tools/),
and Gemma's template may omit some JSON Schema constraints.

<details>
<summary>Native APIs and advanced options</summary>

Share the engine safely from tools, hooks, or application code:

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

Await all work and consume streams inside the callback. Return serializable results to Node.js.
Calls sharing an engine run in order; do not invoke an Agent or another scope on that engine
inside the callback. Direct engine calls bypass the queue. Stop callers before replacing models.

| Native feature                                      | API or limit                                                  |
| --------------------------------------------------- | ------------------------------------------------------------- |
| Raw prompts, echo, multiple choices                 | Native completions; multiple streamed choices are unsupported |
| Generation save and resume                          | Native chat recovery; separate from Strands sessions          |
| Reload, unload, reset, progress, stats, GPU queries | Original engine APIs                                          |
| Logit processors                                    | WebLLM engine setup                                           |
| Cache backends, integrity, custom models            | WebLLM setup and cache APIs                                   |
| Main thread, Worker, Service Worker, extension      | Any `MLCEngineInterface`; use WebLLM's factories and handlers |
| Message `name`                                      | Native chat; SDK messages have no name field                  |

`updateConfig()` merges settings; `modelId` selects an already loaded model.
Use model-card sampling settings. WebLLM does not expose `top_k` or `min_p`.
Cache hints do not configure WebLLM's KV cache. Recovering external tool side effects
is the application's responsibility.

See the [native engine interface](https://github.com/mlc-ai/web-llm/blob/v0.2.86/src/types.ts)
and [Strands provider contract](https://strandsagents.com/docs/user-guide/sdk/model-providers/custom_model_provider/).

</details>
