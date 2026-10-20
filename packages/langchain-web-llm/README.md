# LangChain WebLLM

Run [WebLLM](https://github.com/mlc-ai/web-llm) models locally with WebGPU through LangChain, LangGraph, and Deep Agents.

```sh
npm install langchain-web-llm
```

## Use

Create a worker that runs WebLLM:

```ts
// worker.ts
import { WebWorkerMLCEngineHandler } from '@mlc-ai/web-llm'

const handler = new WebWorkerMLCEngineHandler()
self.onmessage = (message: MessageEvent) => handler.onmessage(message)
```

Load a model in the browser, then pass its engine to the provider:

```ts
import { CreateWebWorkerMLCEngine } from '@mlc-ai/web-llm'
import { ChatWebLLM } from 'langchain-web-llm'

const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
try {
  const engine = await CreateWebWorkerMLCEngine(worker, modelId)
  try {
    const model = new ChatWebLLM({ engine, modelId })
    console.log((await model.invoke('Hello')).text)
  } finally {
    await engine.unload()
  }
} finally {
  worker.terminate()
}
```

The first load fetches WebLLM's runtime and model files; the browser caches them for later use.
Inference runs locally.

Use LangChain's usual `bindTools`, `stream`, `batch`, and `withStructuredOutput` APIs. Call `streamEvents(input)` without a version for typed text, reasoning, tool-call, and usage streams. Thinking is returned as standard `reasoning` content blocks; `.text` contains the answer. Enable it with `request.extra_body.enable_thinking` on a model that supports it.

## Deep Agents

Keep the agent and its tools in Node.js, and WebLLM in a private browser page:

```sh
npm install deepagents
```

```ts
// Browser: after loading the engine
import { createBrowserBridge } from 'langchain-web-llm'
window.webllmLangChainBridge = createBrowserBridge(engine)
```

```ts
// Node.js: page is a caller-owned Playwright Page
import { createDeepAgent } from 'deepagents'
import { BrowserChatWebLLM } from 'langchain-web-llm'

const model = new BrowserChatWebLLM(page, {
  modelId: 'Qwen3.5-4B-q4f16_1-MLC',
  profile: { maxInputTokens: 32768 }, // Match the loaded engine's context window.
  request: { extra_body: { enable_thinking: true } },
})
const agent = createDeepAgent({ model })
const result = await agent.invoke({
  messages: [{ role: 'user', content: 'Write a short greeting to /greeting.txt, then read it.' }],
})
```

Deep Agents writes the text to its virtual filesystem in `result.files`, not to your computer.

Load the browser engine with `context_window_size: 32768` for this example. The page must already contain the loaded model and bridge. [Runnable examples](https://github.com/ykoh42/web-llm-adapters/tree/main/examples/langchain-web-llm) show the full setup.

## Support

| Feature                              | Support                                                                                  |
| ------------------------------------ | ---------------------------------------------------------------------------------------- |
| Chat, streaming, tools, reasoning    | Model-dependent                                                                          |
| Cancellation and shared-engine calls | Yes                                                                                      |
| Structured output                    | `withStructuredOutput()` through tools or native JSON                                    |
| Images and WAV audio                 | Model-dependent; URLs/base64 strings or bytes; no image detail or interleaved text/media |
| Video and generic files              | No native WebLLM input                                                                   |
| Embeddings                           | `WebLLMEmbeddings` and `BrowserWebLLMEmbeddings` for LangGraph memory                    |
| Strict tool schemas                  | No                                                                                       |

For Qwen agents with tools and a response schema, use `responseFormat: toolStrategy(schema)` ([`toolStrategy`](https://reference.langchain.com/javascript/langchain/index/toolStrategy) from `langchain`); native JSON cannot be combined with Qwen tools.

Use literal `~0`/`~1` in `$ref` paths; LangChain's tool validator cannot resolve `%7E0`/`%7E1`.

## Models

These exact WebLLM IDs passed real-model checks. Tools includes default Deep Agents → tool result → final answer. `O` = verified; `—` = unavailable for this artifact.

| WebLLM model ID                         | Tools | Thinking | Images | Embeddings |
| --------------------------------------- | ----- | -------- | ------ | ---------- |
| `Qwen3.5-4B-q4f16_1-MLC`                | O     | O        | —      | —          |
| `Qwen3-8B-q4f16_1-MLC`                  | O     | O        | —      | —          |
| `gemma-4-E2B-it-q4f16_1-MLC`            | O     | O        | —      | —          |
| `Qwen2.5-3B-Instruct-q4f16_1-MLC`       | O     | —        | —      | —          |
| `Phi-3.5-vision-instruct-q4f16_1-MLC`   | —     | —        | O      | —          |
| `snowflake-arctic-embed-s-q0f32-MLC-b4` | —     | —        | —      | O          |

Gemma 4 E2B needs the bundled [runtime setup and protocol limits](https://github.com/ykoh42/web-llm-adapters/blob/main/shared/runtime/README.md). Its tools and thinking accept text and tool results only. Model-selected tools and arguments still need validation.

## Embeddings

Load an embedding model, then use LangChain's standard interface:

```ts
import { WebLLMEmbeddings } from 'langchain-web-llm'

const embeddings = new WebLLMEmbeddings({
  engine,
  modelId: 'snowflake-arctic-embed-s-q0f32-MLC-b4',
})
const vector = await embeddings.embedQuery('a flower')
```

Snowflake's query prefix is applied automatically; documents stay unprefixed. Use `BrowserWebLLMEmbeddings(page, options)` for Node.js and LangGraph semantic memory.
Document inputs are sent in batches of 32 by default; lower `batchSize` to reduce temporary memory use. With `maxRetries`, only the failed batch is retried.

## Native options

Set generation options with `request` at construction or per call. `onChunk` exposes unchanged native chunks, including log probabilities and timing.

Chat retries default to 0 and apply only while opening a stream. Context-limit errors use LangChain's `ContextOverflowError`, with the native cause preserved.

For custom vision models in Deep Agents, set `profile.imageInputs` and `profile.imageToolMessage` to `true`. Tool images are sent in the next user turn.

Use `model.engine` and `withEngine(engine, callback)` for other native APIs; consume streams inside the callback. You own model loading, the engine, workers, and browser page. Native calls on one engine run in order. Cancellation interrupts generation without unloading the model.
