# strands-web-llm

A community WebLLM provider for Strands SDK and Harness. Run inference in a
WebGPU browser while tools run in your application.

## Install

```sh
npm install strands-web-llm
```

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
npm install @strands-agents/harness@0.3.0
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

## Models

Exact WebLLM **0.2.86** IDs checked in Harness. `O` = verified; `△` = unreliable;
`X` = unverified. Tools covers call → result → final answer; Think means separate reasoning blocks.
Decode speeds are local measurements on M4 Pro (24 GB), Chrome 155.

| Exact WebLLM `model_id`             |                                                                                 MMLU-Pro | Decode tok/s | Tools | Think | Image | Audio | Video |
| ----------------------------------- | ---------------------------------------------------------------------------------------: | -----------: | ----- | ----- | ----- | ----- | ----- |
| `Qwen3.5-9B-q4f16_1-MLC`            |                                           [82.5](https://huggingface.co/Qwen/Qwen3.5-9B) |         19.7 | O     | O     | X     | X     | X     |
| `Qwen3.5-4B-q4f16_1-MLC`            |                                           [79.1](https://huggingface.co/Qwen/Qwen3.5-4B) |         33.0 | O     | O     | X     | X     | X     |
| `Qwen3-8B-q4f16_1-MLC`              |                 [74.2](https://huggingface.co/nvidia/Qwen3-8B-DMS-8x#evaluation-dataset) |         32.0 | O     | O     | X     | X     | X     |
| `Qwen3-4B-q4f16_1-MLC`              |                   [70.4](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507#performance) |         51.5 | △     | O     | X     | X     | X     |
| `Qwen3.5-2B-q4f16_1-MLC`            |                                           [66.5](https://huggingface.co/Qwen/Qwen3.5-2B) |         59.7 | O     | O     | X     | X     | X     |
| `gemma-4-E2B-it-q4f16_1-MLC`        |                                     [60.0](https://huggingface.co/google/gemma-4-E2B-it) |         38.5 | O     | O     | X     | X     | X     |
| `Qwen2.5-7B-Instruct-q4f16_1-MLC`   |                                       [56.3](https://qwenlm.github.io/blog/qwen2.5-llm/) |         38.0 | O     | X     | X     | X     | X     |
| `Qwen2.5-3B-Instruct-q4f16_1-MLC`   |                                       [43.7](https://qwenlm.github.io/blog/qwen2.5-llm/) |         59.5 | O     | X     | X     | X     | X     |
| `Qwen3.5-0.8B-q4f16_1-MLC`          |                                         [42.3](https://huggingface.co/Qwen/Qwen3.5-0.8B) |         71.2 | O     | O     | X     | X     | X     |
| `Qwen3-0.6B-q4f16_1-MLC`            | [38.27](https://huggingface.co/MultiverseComputingCAI/LittleLamb#evaluation--benchmarks) |         90.4 | △     | O     | X     | X     | X     |
| `Qwen2.5-1.5B-Instruct-q4f16_1-MLC` |                                       [32.4](https://qwenlm.github.io/blog/qwen2.5-llm/) |         87.6 | O     | X     | X     | X     | X     |
| `Qwen2.5-0.5B-Instruct-q4f16_1-MLC` |                                       [15.0](https://qwenlm.github.io/blog/qwen2.5-llm/) |        136.8 | O     | X     | X     | X     | X     |

Qwen3 0.6B can miscopy arguments or omit answers with thinking on;
Qwen3 4B can call dependent tools together.
Gemma 4 E2B requires the bundled [runtime setup and protocol limits](https://github.com/ykoh42/web-llm-adapters/blob/main/shared/runtime/README.md).

<details>
<summary>Benchmark notes and tool formats</summary>

Models are ranked by published upstream MMLU-Pro with thinking on; Qwen2.5 has no thinking mode.
Sources and evaluation settings vary. Decode speeds are the median of three local runs,
thinking off, max 128 tokens; they are not portable across devices.

Tool formats follow [Hermes](https://github.com/QwenLM/Qwen3/blob/main/docs/source/framework/function_call.md)
(Qwen2.5/Qwen3), [Qwen3 Coder](https://github.com/vllm-project/vllm/blob/5c1c903501927ea7c6c699aabd4c271015e66e71/vllm/parser/qwen3.py)
(Qwen3.5), and [Gemma](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4).

</details>

### Images and embeddings

| Exact WebLLM `model_id`                 | Strands API                      | Verified use                       |
| --------------------------------------- | -------------------------------- | ---------------------------------- |
| `Phi-3.5-vision-instruct-q4f16_1-MLC`   | `ImageBlock`                     | Harness reads an image             |
| `snowflake-arctic-embed-s-q0f32-MLC-b4` | `MemoryStore` / `SearchStrategy` | Semantic search and Harness memory |

`WebLLMEmbeddingSearchStrategy` connects WebLLM to Strands memory; see the
[memory example](https://github.com/ykoh42/web-llm-adapters/blob/main/examples/strands-web-llm/harness/src/memory.ts).
Arctic Embed S has 384 dimensions and a 512-token input limit. Prefix queries with
`Represent this sentence for searching relevant passages: `; leave stored text unprefixed.
Search returns positive cosine matches by default; set `minScore` to include lower scores.
For FileMemoryStore, set `includeData: false` to avoid retaining bytes it reads back itself.
Searches scan all entries, and vectors rebuild after restart. Keep each stored record within 512 tokens;
FileMemoryStore appends facts that share a heading.

Other vision and embedding IDs and speech understanding are unverified. Video is unsupported.

## Support

Targets: WebLLM **0.2.86**, Strands SDK **1.21.0**, Harness **0.3.0**.
Features depend on the exact model above.

| Feature                               | Use                                               | Limit                                                    |
| ------------------------------------- | ------------------------------------------------- | -------------------------------------------------------- |
| Chat, system prompts, history         | Strands messages and `systemPrompt`               | Unsupported blocks raise errors                          |
| Streaming and cancellation            | `agent.stream()`, `agent.cancel()`                | One assistant response per call                          |
| Tools and results                     | Strands tools and `toolChoice`                    | Qwen supports `any`; Gemma and native tool APIs do not   |
| Disable tools                         | `request.tool_choice: 'none'`                     | Keeps the Agent tool registry                            |
| Thinking                              | `request.extra_body.enable_thinking`              | Render `textBlock` values to show only the answer        |
| Sampling and token limits             | `request`; `maxTokens`, `temperature`, `topP`     | SDK aliases take precedence                              |
| JSON schema, grammar, structural tags | `request.response_format`                         | Cannot combine with Qwen tools or Gemma protocol mode    |
| Images and WAV audio                  | `ImageBlock`, `AudioBlock`, `pcmAudio()`          | Model-dependent; PCM is converted to WAV                 |
| Usage, logprobs, timing               | Strands metadata and `onChunk`                    | Native chunk fields retained                             |
| Embeddings                            | `WebLLMEmbeddingSearchStrategy` or `withEngine()` | Separate embedding model required                        |
| Context recovery                      | `countTokens()` and Strands overflow error        | Counts are estimates; set the loaded context window size |
| Other native APIs                     | `withEngine()`                                    | Caller owns setup and lifecycle                          |

Signed/redacted reasoning, S3 media, video, documents, guardrails, and non-image tool-result media
are unsupported. Tool-result images follow the adjacent tool replies; no listed model is verified
for both tools and vision. Invalid or incomplete tool output reaches the caller; the adapter does not
change tool names or arguments or supply answers. Model-selected tools and arguments still need validation.

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

[MIT](LICENSE). The bundled runtime retains its [upstream licenses and notices](https://github.com/ykoh42/web-llm-adapters/blob/main/shared/runtime/NOTICE.txt).
