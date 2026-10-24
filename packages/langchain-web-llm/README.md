# LangChain WebLLM

Run [WebLLM](https://github.com/mlc-ai/web-llm) models locally with WebGPU through LangChain, LangGraph, and Deep Agents.

```sh
npm install langchain-web-llm
```

- Yarn: also install `@mlc-ai/web-llm` and `@langchain/core`. Check compatible versions with `yarn explain peer-requirements`.
- pnpm: if dependency scripts are blocked, run [`pnpm approve-builds`](https://pnpm.io/cli/approve-builds) to allow or deny them, then rerun `pnpm install`.

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

Use LangChain's usual `bindTools`, `stream`, `batch`, and `withStructuredOutput` APIs. For `strict: true`, use `method: 'jsonSchema'`. Call `streamEvents(input)` without a version for typed text, reasoning, tool-call, and usage streams. Thinking is returned as standard `reasoning` content blocks; `.text` contains the answer. Enable it with `request.extra_body.enable_thinking` on a model that supports it.

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

| Feature                              | Support                                                                                                        |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| Chat, streaming, tools, reasoning    | Model-dependent                                                                                                |
| Required tool choice                 | Qwen: one or more tools; Gemma/native tool models: one                                                         |
| Cancellation and shared-engine calls | Yes                                                                                                            |
| Token counts                         | LangChain estimates (`getNumTokens`)                                                                           |
| LangChain response cache (`cache`)   | Known upstream cache defects; leave unset ([#10092](https://github.com/langchain-ai/langchainjs/issues/10092)) |
| Structured output                    | `withStructuredOutput()` through tools or native JSON                                                          |
| Images                               | Model-dependent; URLs/base64/bytes; no detail                                                                  |
| WAV and Float32 PCM audio            | Model-dependent; WAV base64/bytes/data URLs or native `pcm_f32`                                                |
| Video and generic files              | No native WebLLM input                                                                                         |
| Embeddings                           | `WebLLMEmbeddings` and `BrowserWebLLMEmbeddings` for LangGraph memory                                          |
| Strict tool schemas                  | No                                                                                                             |

Keep text blocks together in media messages. Browser transport carries PCM as float WAV without changing samples or sample rate.

Tested with LangChain Core **1.2.8** and **1.3.0**, and WebLLM **0.2.79** and **0.2.86**.

For Qwen agents with tools and a response schema, use `responseFormat: toolStrategy(schema)` ([`toolStrategy`](https://reference.langchain.com/javascript/langchain/index/toolStrategy) from `langchain`); native JSON cannot be combined with Qwen tools.

Use Gemma agent response schemas without other tools.

Use literal `~0`/`~1` in `$ref` paths; LangChain's tool validator cannot resolve `%7E0`/`%7E1`.

## Models

These exact WebLLM IDs passed real-model checks. Tools includes default Deep Agents → tool result → final answer. `O` = verified; `—` = unavailable for this artifact.

| WebLLM model ID                          | Tools | Thinking | Images | Audio | Embeddings |
| ---------------------------------------- | ----- | -------- | ------ | ----- | ---------- |
| `Qwen3.5-4B-q4f16_1-MLC`                 | O     | O        | —      | —     | —          |
| `Qwen3-8B-q4f16_1-MLC`                   | O     | O        | —      | —     | —          |
| `gemma-4-E2B-it-q4f16_1-MLC`             | O     | O        | —      | O     | —          |
| `Qwen2.5-3B-Instruct-q4f16_1-MLC`        | O     | —        | —      | —     | —          |
| `Phi-3.5-vision-instruct-q4f16_1-MLC`    | —     | —        | O      | —     | —          |
| `snowflake-arctic-embed-m-q0f32-MLC-b32` | —     | —        | —      | —     | O          |
| `snowflake-arctic-embed-m-q0f32-MLC-b4`  | —     | —        | —      | —     | O          |
| `snowflake-arctic-embed-s-q0f32-MLC-b32` | —     | —        | —      | —     | O          |
| `snowflake-arctic-embed-s-q0f32-MLC-b4`  | —     | —        | —      | —     | O          |

Install WebLLM `0.2.79` or later in the `0.2.x` series. Gemma 4 E2B was verified with WebLLM `0.2.86` and needs the bundled [runtime setup and protocol limits](https://github.com/ykoh42/web-llm-adapters/blob/main/shared/runtime/README.md). Audio works in ordinary chat; the default profile keeps it disabled for Deep Agents. Tools and thinking accept text and tool results only. Model-selected tools and arguments still need validation.

For Qwen3.5, declare object/array types to preserve XML in JSON. History cannot preserve `>` in top-level argument names or `<parameter=` / `</parameter>` in raw strings.

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

Arctic Embed M returns 768 dimensions; S returns 384. Both have a 512-token input limit.
Snowflake's query prefix is applied automatically; documents stay unprefixed. Use `BrowserWebLLMEmbeddings(page, options)` for Node.js and LangGraph semantic memory.
Document inputs are sent in batches of 32 by default; lower `batchSize` to reduce temporary memory use. With `maxRetries`, only the failed batch is retried.

## Native options

Set generation options with `request` at construction or per call. For probabilities, set `logprobs: true` and `top_logprobs: 2`; read `response_metadata.logprobs`. `onChunk` exposes unchanged native chunks and timing; `BrowserChatWebLLM` sends them across the page bridge.

For Gemma 4, `request.extra_body.preserve_thinking: true` also keeps earlier tool-call reasoning; the default keeps only the current user turn.

Set `thinkingPrefilled: true` at construction if your loaded Qwen template already opens `<think>`. This only controls output parsing; it does not change the engine or prompt.

Chat retries default to 0 and apply only while opening a stream. Context-limit errors use LangChain's `ContextOverflowError`, with the native cause preserved.

`BrowserChatWebLLM` awaits `onFailedAttempt` in Node.js before retrying; throwing stops the call. Errors preserve JSON fields and cause values, plus LangChain retry markers.

For custom vision models in Deep Agents, set `profile.imageInputs` and `profile.imageToolMessage` to `true`. Tool images are sent in the next user turn.

Use `withEngine(model.engine, callback)` for native APIs, or `window.webllmLangChainBridge.withEngine(callback)` inside the browser page. Finish all work and consume streams inside the callback. Calls on one engine run in order. You own model loading, the engine, workers, and page; cancellation does not unload them.
