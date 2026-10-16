# strands-webllm

A community-maintained WebLLM model provider for Strands Harness and its SDK.
The [Harness example](../../examples/harness/README.md) connects Node.js Harness
to browser inference; this provider itself runs beside the WebLLM engine.

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

Use `agent.cancel()` to interrupt generation. Do not run native engine calls or
change loaded models while an agent call is active. Overlapping adapter calls on
the same engine are rejected so one call cannot cancel another.

## Native features

`model.engine` is the same engine you pass in. Use WebLLM's own API for loading,
cache management, embeddings, text completions, and other engine features.

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
outer iterator suspended. The next call waits for that cleanup. Other overlapping
adapter calls are rejected. Do not mix native calls or model reloads with an
active adapter call.

## Limits

WebLLM 0.2.86 accepts native tool calls only on its listed Hermes models. Gemma
4 E2B handles text and audio; use Phi vision for images. Hermes tool mode supplies
its own system prompt and JSON grammar. Set Harness `systemPrompt: ''` to disable
its prompt for this mode. Empty prompts are omitted by the adapter. The tool example
ends after the tool batch because Hermes retains the tool grammar on subsequent calls.
Strands `toolChoice.any` has no native WebLLM equivalent and is rejected.

Supported media use image URLs/bytes or WAV bytes. Raw PCM uses the native engine.
S3 media, video, documents, guardrails, reasoning blocks, and media in tool results
raise explicit errors. Text blocks are joined into WebLLM's single text part.
Cache-point hints do not configure the native KV cache. Qwen's thinking toggle
controls generated thinking text; it does not create Strands reasoning blocks.

Multiple choices, non-streaming requests, text completion, embeddings, cache,
loading, token-level sampling, and save/resume keep WebLLM's native APIs. Native
persistence does not restore a Strands Agent, its tools, or its history. Our resume
example checks an orderly interruption. Shared cross-origin cache needs its separate
extension; this example uses it on the main thread and disables unsupported deletion.

The [browser examples](../../examples/browser/README.md) and their executable GPU
tests demonstrate each feature group. Native access does not imply that every
feature has a Strands counterpart or that every model and device is supported.
