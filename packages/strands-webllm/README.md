# strands-webllm

A community-maintained WebLLM model provider for Strands Harness and its SDK.
Use `WebLLMModel` beside the engine, or `BrowserModel` to connect Node.js Harness
to a browser. Both use the same provider and inference queue.

## Use

This package is not published yet. Run the browser example from the repository.
Once published, install `strands-webllm`, `@strands-agents/sdk`, and
`@mlc-ai/web-llm` together.

```ts
import { CreateMLCEngine } from '@mlc-ai/web-llm'
import { Agent } from '@strands-agents/sdk'
import { WebLLMModel } from 'strands-webllm'

const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
const engine = await CreateMLCEngine(
  modelId,
  {
    initProgressCallback: (report) => console.log(report.text),
  },
  { context_window_size: 2048 },
)

const model = new WebLLMModel({
  engine,
  modelId,
  contextWindowLimit: 2048,
  maxTokens: 256,
  temperature: 0,
})
const agent = new Agent({ model })
const result = await agent.invoke('Say hello in one sentence.')
console.log(result.lastMessage)

await engine.unload()
```

Load the engine in a WebGPU-capable browser. The adapter does not access browser
globals when imported, load a model, or unload an engine that the caller owns.

## Node.js Harness

Load WebLLM on a page you control, then expose the package's bridge on that page:

```ts
import { createBrowserBridge } from 'strands-webllm'

// engine is a loaded WebLLM engine, created by your browser application.
window.webllmBridge = createBrowserBridge(engine)
```

Pass that page to the public `BrowserModel` in Node.js:

```ts
import { createHarness } from '@strands-agents/harness'
import { BrowserModel } from 'strands-webllm'

// page is a Playwright Page opened to your browser application.
const model = new BrowserModel(page, {
  modelId: 'gemma-4-E2B-it-q4f16_1-MLC',
  contextWindowLimit: 16384, // Match the engine's loaded context window.
  maxTokens: 4096,
  request: { extra_body: { enable_thinking: true } },
  onChunk: (chunk) => console.log(chunk), // Optional native chunks, including logprobs.
})
const agent = await createHarness({ model })
try {
  const result = await agent.invoke('Say hello.')
  console.log(result.lastMessage.content)
} finally {
  await agent.shutdown()
}
```

Harness runs tools in Node.js; the bridge returns Strands stream events from the
browser. Thinking, tool choice, trailing Harness context, cancellation, and
context overflow errors pass through this connection. SDK message JSON preserves
media bytes. Model support and Gemma protocol restrictions still apply.

`onChunk` runs in Node.js. Original chunks arrive in order with the next Strands
event or error; they can be batched while the provider buffers a tool call. They
are transferred only when a callback is supplied. A callback error stops the call
and releases the engine. The callback is not part of the serializable model config.

`BrowserPage` requires only `evaluate()` and accepts a Playwright Page without a
Playwright dependency in this package. The application owns browser startup,
engine loading, and shutdown. Before unloading the engine or closing the page,
run `await page.evaluate(() => window.webllmBridge.close())`. This stops model calls, waits for active native callbacks to finish, and rejects
new requests and queued native callbacks. It does not unload the engine.
Do not replace a bridge while its calls are running, and expose it only on a
trusted page controlled by your application.

The [complete Harness example](../../examples/harness/README.md) sets up Chrome,
the browser page, and optional read, search, memory, and summary behavior. Those
application choices stay in the example; no example source is needed to import
the package's connection API.

## Streaming

```ts
for await (const update of agent.stream('Say hello.')) {
  if (update.type !== 'modelStreamUpdateEvent') continue
  const event = update.event
  if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta') {
    console.log(event.delta.text)
  }
}
```

Use `agent.cancel()` to interrupt generation. Adapter calls and `withEngine()`
callbacks sharing one engine wait in order. Cancelling a waiting call does not interrupt the active call.
Separate engines have separate queues.

## Native features

`model.engine` is the same engine you pass in. `withEngine()` gives native work
its own turn in the adapter's queue. Use it in Strands tools and hooks for APIs
outside the SDK's `Model` contract, including embeddings, raw completions,
multiple responses, and generation checkpoints. The callback receives the original
`MLCEngineInterface`; there is no method whitelist or replacement response format.

```ts
import { withEngine } from 'strands-webllm'

// Load an embedding model with WebLLM before calling this tool callback.
const vector = await withEngine(engine, async (engine) => {
  const result = await engine.embeddings.create({
    model: 'snowflake-arctic-embed-s-q0f32-MLC-b4',
    input: 'hello',
  })
  return result.data[0]!.embedding
})
```

Node.js tools use the same scope through the browser bridge and Playwright:

```ts
const vector = await page.evaluate(
  (input) =>
    window.webllmBridge.withEngine(async (engine) => {
      const result = await engine.embeddings.create({
        model: 'snowflake-arctic-embed-s-q0f32-MLC-b4',
        input,
      })
      return result.data[0]!.embedding
    }),
  'hello',
)
```

The callback runs on the browser page. Pass inputs through `evaluate()` and return
serializable results; Node.js closures do not travel to the browser. Return the
vector from a Strands tool callback to give it to the agent. The
[executable Harness test](../../examples/harness/test/engine.browser.ts) registers
such a tool and runs it alongside another native request.

Await all native work and consume streams **inside** the callback. Returning an
async iterable is rejected and closes it before releasing the engine. Do not call
an Agent, an adapter model, or another `withEngine()` on the same engine inside
the callback: that operation would wait for its own turn to end. Cancel native
generation with WebLLM's `interruptGenerate()`; `agent.cancel()` controls only the
agent's generation. The caller still owns model loading and shutdown.

Use this scope for engine mutations too. Before replacing the loaded model set,
stop callers that depend on it and update their model configs. The queue prevents
overlapping engine work; it does not rewrite requests already waiting for an old
model. Direct engine calls bypass this queue. Module-level cache APIs and worker
setup remain WebLLM's own APIs:

```ts
import { hasModelInCache, deleteModelAllInfoInCache } from '@mlc-ai/web-llm'

const cached = await hasModelInCache(modelId)
console.log(cached)

// After unloading the model, when the app wants to remove its cached files:
await deleteModelAllInfoInCache(modelId)
```

Pass a `WebWorkerMLCEngine`, `ServiceWorkerMLCEngine`, or
`ExtensionServiceWorkerMLCEngine` in the same way. Workers
must be created and connected with WebLLM's own factories and handlers.

Generation options that Strands does not name go in `request`. To keep native
chunks, including log probabilities and extra timing data, use `onChunk`.
Its `WebLLMChunk` type is a union: `object === 'text_completion'` identifies a raw
Gemma tool-mode chunk; other requests produce `chat.completion.chunk` chunks.

```ts
const model = new WebLLMModel({
  engine,
  modelId,
  request: {
    seed: 42,
    repetition_penalty: 1.1,
    response_format: { type: 'json_object' },
  },
  onChunk: (chunk) => console.log(chunk),
})
```

For JSON output, also ask the model for JSON in the prompt. Strands controls
messages, tools, streaming, model selection, and a single response per request.
Use the native engine for options that change that contract, such as multiple
responses or non-streaming completions. Native features remain accessible; that
does not mean they are all mapped into Strands.

`updateConfig()` merges settings. Updating `modelId` selects an already loaded
model. Call `engine.reload()` first to load a new one, and set a matching
`contextWindowLimit` if the context size changes.

## Support

Target versions: Strands SDK **1.21.0**, WebLLM **0.2.86**.

Text, system prompts, streaming, usage, tools, JSON settings, image bytes/URLs,
and WAV bytes map to Strands. Each feature still requires a compatible WebLLM
model. Other WebLLM operations keep their native API through `model.engine`.

The adapter closes the native stream on cancellation even if Strands leaves an
outer iterator suspended. The next call waits for that cleanup. Once the native
stream finishes, hooks and other agents can use the engine while Strands finishes
processing its events. Use `withEngine()` to coordinate native calls with these
model calls.

Gemma token counts omit past-turn thoughts just as the generated prompt does,
while keeping thoughts from the current tool loop. Counts estimate declarations
in their native form. They remain estimates; native context overflow is
translated to Strands's recovery error,
including errors returned as strings by a WebLLM Worker. Set `contextWindowLimit`
to the window used when loading the engine.

## WebLLM API coverage

The compatibility goal is to preserve everything the two libraries support:
request settings, input data, output data, errors, and engine behavior. Adapter
work closes losses or unnecessary restrictions in that connection. Features keep
their recommended API: Strands model turns use `Model`; other engine operations
use WebLLM directly through the shared queue. Upstream restrictions are documented
as boundaries of the supported versions.

This inventory targets WebLLM **0.2.86** and Strands SDK **1.21.0**. Both model
classes support the mapped generation features below. “Native scope” means
`withEngine(engine, callback)` in the browser or `window.webllmBridge.withEngine(callback)`
from a Node.js tool through `page.evaluate()`. It coordinates original WebLLM APIs
with Strands; it does not change the SDK Model contract.

| WebLLM surface                                                                                                           | Strands connection                                                            | Boundary                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Chat messages, system prompt, tool definitions and results                                                               | Agent messages, `systemPrompt`, and registered tools                          | Unsupported Strands block types raise errors                                             |
| `model`                                                                                                                  | `modelId`                                                                     | Selects an already loaded model                                                          |
| `max_tokens`, `temperature`, `top_p`                                                                                     | `request`, or SDK `maxTokens`, `temperature`, `topP`                          | SDK settings take precedence                                                             |
| `frequency_penalty`, `presence_penalty`, `repetition_penalty`, `seed`, `stop`, `logit_bias`                              | `request`                                                                     | Passed to the native request                                                             |
| `logprobs`, `top_logprobs`                                                                                               | `request` and `onChunk`                                                       | Original probabilities available in both model classes                                   |
| Usage, timing, latency breakdown                                                                                         | Strands usage/trace, `onChunk`, `request.extra_body.enable_latency_breakdown` | Native extra fields retained                                                             |
| `response_format`: text, JSON schema, grammar, structural tags                                                           | `request.response_format`                                                     | Native chat supports constraints; Gemma raw mode accepts plain text                      |
| `tool_choice`: auto, named, none                                                                                         | SDK `toolChoice`; `request.tool_choice: 'none'`                               | An explicit SDK choice overrides the config; native tools are model-restricted           |
| Image URL, data URL and bytes                                                                                            | SDK `ImageBlock`                                                              | Requires a vision model; Gemma raw protocol is text-only                                 |
| User/assistant message `name`                                                                                            | Native scope                                                                  | SDK messages have no corresponding name field                                            |
| WAV and float32 PCM audio                                                                                                | SDK `AudioBlock`; `pcmAudio(samples, sampleRate)`                             | PCM is carried losslessly in float WAV; requires an audio model                          |
| Thinking toggle                                                                                                          | `request.extra_body.enable_thinking`                                          | Gemma reasoning is separated; other native models retain their own output format         |
| `ignore_eos`                                                                                                             | `request.ignore_eos` on native chat                                           | Gemma protocol rejects it to preserve turn boundaries                                    |
| Streaming and non-streaming chat consumption                                                                             | `agent.stream()` and `agent.invoke()`                                         | Both consume one native stream; no independent non-streaming backend mode                |
| Multiple choices (`n > 1`)                                                                                               | Native scope                                                                  | Strands Model produces one assistant turn; WebLLM also rejects multiple streamed choices |
| Crash-resumable request configuration                                                                                    | `request.extra_body.resumable` on native chat                                 | Listing/recovery/deletion use native scope; Agent/tool state is separate                 |
| Raw completions (`prompt`, `echo`, multiple non-streaming choices)                                                       | Native scope                                                                  | Use the original completion API inside a tool or hook                                    |
| Embeddings                                                                                                               | Native scope                                                                  | Use the original embedding API inside a tool or hook                                     |
| Reload/unload, app config, progress callbacks, reset, saved message, stats, GPU queries, log level, token-level sampling | Caller-owned engine                                                           | Coordinate engine work through native scope                                              |
| Cache backends, cache inspection/deletion, integrity, custom models and logit processors                                 | Caller-owned WebLLM setup                                                     | Pass the configured engine; the adapter does not recreate it                             |
| Main thread, Web Worker, Service Worker, extension worker                                                                | Pass any `MLCEngineInterface`                                                 | BrowserModel additionally carries the provider to Node.js                                |

Every method on the original engine is available in native scope, including from
Node.js Harness tools. This is API access, not a claim that every API combination
has been tested. Real GPU coverage includes an embedding tool with Gemma, concurrent
raw multiple completions, orderly interruption, and recovery after destroying the
page during generation. The recovery test uses exact persistence, recreates the
engine, resumes the native checkpoint, and separately restores a saved Harness
conversation with its reasoning and completed tool results.

Native generation checkpoints do not restore a Strands Agent, its tool execution,
or its history. Use Strands session persistence for that state and WebLLM's own
recovery API for native generation. Automatic recovery of a partially executed
Agent turn is not provided. A tool interrupted after its external effect but
before its result was saved needs application-level idempotency or reconciliation;
replaying model tokens cannot establish whether that effect happened. Gemma media
plus tools/thinking, constrained output in the raw protocol, and `ignore_eos` with protocol parsing remain unavailable
combinations. No fallback model or prompt instruction substitutes for them.

This separation follows Strands's [custom model provider contract](https://strandsagents.com/docs/user-guide/sdk/model-providers/custom_model_provider/)
and WebLLM's [native usage](https://webllm.mlc.ai/docs/user/basic_usage.html):
the adapter maps model turns, while tools and application code use the remaining
engine capabilities directly.

The distinction follows the native [chat request](https://github.com/mlc-ai/web-llm/blob/v0.2.86/src/openai_api_protocols/chat_completion.ts),
[completion request](https://github.com/mlc-ai/web-llm/blob/v0.2.86/src/openai_api_protocols/completion.ts),
and [engine interface](https://github.com/mlc-ai/web-llm/blob/v0.2.86/src/types.ts).
WebLLM rejects `image_url.detail` even though its type declares that field.
Raw completions explicitly reject resumable configuration. Native availability
does not imply that all combinations are accepted by WebLLM or the loaded artifact.

## PCM audio

```ts
import { pcmAudio } from 'strands-webllm'
import { TextBlock } from '@strands-agents/sdk'

// Mono Float32Array, for example from AudioBuffer.getChannelData(0).
const audio = pcmAudio(samples, sampleRate)
await agent.invoke([new TextBlock('Transcribe this audio.'), audio])
```

The helper returns a normal SDK `AudioBlock`. It preserves float32 samples and the
sample rate in an IEEE float WAV, so storage and the browser bridge need no custom
message type. It does not resample, downmix, or quantize. Mix multichannel input to
mono before calling it. As with WAV input, omit the explicit Gemma thinking flag
and tools to use its native audio path.

## Gemma thinking

Enable thinking with the existing request option:

```ts
const model = new WebLLMModel({
  engine,
  modelId: 'gemma-4-E2B-it-q4f16_1-MLC',
  maxTokens: 1024,
  request: { extra_body: { enable_thinking: true } },
})
```

The adapter uses Gemma's raw prompt protocol with or without tools. Thinking
streams as `reasoningContentDelta` and is stored in Strands `ReasoningBlock`
objects; answers remain text blocks. `onChunk` still receives the original raw
WebLLM chunks, including control tokens.

Following [Google's thinking rules](https://ai.google.dev/gemma/docs/capabilities/thinking),
thoughts stay in the prompt throughout the current tool loop. A new user turn
removes older thoughts from the next model prompt. The Agent's stored messages
are not changed. Disabling thinking does not discard thoughts from a tool loop
that is still in progress. Signed and redacted reasoning are rejected.

To show only the final answer, select text blocks from `result.lastMessage.content`.
Strands `String(result)` includes reasoning too. The browser examples put reasoning
in **Details** and only answer text in **Response**. Thinking shares the output
token limit, so allow more tokens than you would for a short plain answer.

## Limits

Gemma 4 models with IDs starting with `gemma-4-` use the adapter's Gemma protocol
when tools are registered. This follows Google's
[native function-call template](https://huggingface.co/google/gemma-4-E2B-it/blob/main/chat_template.jinja)
through WebLLM raw completions. The model selects a tool, Strands or Harness runs
it, and the result returns to Gemma for another call or a final answer. Calls are
validated before execution; incomplete or truncated calls do not execute tools.
The real GPU tests use `gemma-4-E2B-it-q4f16_1-MLC`.

Gemma protocol mode supports text, JSON tool arguments/results, system prompts, auto
or named tool choice, sampling, usage, cancellation, and streamed answer text.
Use `request: { tool_choice: 'none' }` to omit tool declarations while keeping the
Agent's registry. An explicit SDK `toolChoice` for a call takes precedence.
Named choice prefills the selected call and its first required field; the model
still supplies the values. Structured JSON and multiple result blocks retain
their structure. Harness reminders and background continuations stay inside the
current tool turn without changing the stored conversation.
It accepts `response_format: { type: 'text' }` or an empty format with tools and
thinking. Plain text is already the raw protocol's output mode. It rejects media,
constrained output formats, enabled resume, and `ignore_eos`.
`extra_body.resumable: { enabled: false }` is accepted and does not enable
persistence on the raw completion path.
It rebuilds the raw prompt on each call, so it does not reuse chat KV state.
Use the standard WebLLM Gemma 4 catalog records, whose artifact loader supplies BOS;
custom aliases and older Gemma templates are not covered. Gemma uses WebLLM's
normal chat API when there are no tools, no explicit thinking flag, and no reasoning
from the current tool loop. Past-turn thoughts are removed before choosing the
request path, so a later user turn can return to native JSON or audio without
clearing Agent history. The native path includes text and audio support.

WebLLM 0.2.86's native `tools` API accepts only its listed Hermes models. That path
remains available: Hermes supplies its own system prompt and JSON grammar. Set
Harness `systemPrompt: ''` for this legacy mode. The Hermes example ends after the
tool batch because Hermes retains the tool grammar on subsequent calls.
Strands `toolChoice.any` is rejected in both modes. Use Phi vision for images.

Supported media use image URLs/bytes, WAV bytes, or PCM through `pcmAudio()`.
S3 media, video, documents, guardrails, and media in tool results
raise explicit errors. Text blocks are joined into WebLLM's single text part.
Reasoning blocks are supported only by the Gemma protocol path.
Cache-point hints do not configure the native KV cache. Qwen's thinking toggle
controls generated thinking text; it does not create Strands reasoning blocks.

Multiple choices, non-streaming requests, text completion, embeddings, cache,
loading, token-level sampling, and save/resume keep WebLLM's native APIs. Native
persistence does not restore a Strands Agent, its tools, or its history. Our resume
example checks an orderly interruption; the Harness recovery test also destroys
the page during an active generation and restores its saved checkpoint.
Shared cross-origin cache needs its separate extension; this example uses it on the main thread and disables unsupported deletion.

The [browser examples](../../examples/browser/README.md) and their executable GPU
tests demonstrate each feature group. Native access does not imply that every
feature has a Strands counterpart or that every model and device is supported.
