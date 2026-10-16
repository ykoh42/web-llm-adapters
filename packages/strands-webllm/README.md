# strands-webllm

A community-maintained WebLLM model provider for Strands Agents.

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

Pass a `WebWorkerMLCEngine` or `ServiceWorkerMLCEngine` in the same way. Workers
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

| Feature                                      | Strands adapter                             | Check                                |
| -------------------------------------------- | ------------------------------------------- | ------------------------------------ |
| Text, system prompt, streaming               | Supported                                   | Unit tests and Gemma browser test    |
| Tool calls and results                       | Mapped to native WebLLM tools               | Agent loop with a mocked engine      |
| Stop reason, token counts, cancel            | Supported                                   | Unit tests                           |
| JSON output, sampling options                | Passed to WebLLM                            | Request mapping tests                |
| Image URLs and bytes                         | User messages only; requires a vision model | Message mapping tests                |
| WAV bytes                                    | User messages only; requires an audio model | Message mapping tests                |
| Progress and cache management                | Native WebLLM API                           | Browser load; native API access      |
| Web Workers and Service Workers              | Accept their native engine interface        | Not yet tested in a worker           |
| Embeddings, text completion, resume, raw PCM | Native engine API                           | Not mapped or tested by this adapter |

WebLLM 0.2.86 only accepts native tools for its listed Hermes models. Gemma 4 E2B
supports text and audio in this release, but is not in that tool list. The adapter
does not add prompt-based tool parsing. WebLLM's Hermes tool mode also rejects
custom system prompts and custom response formats. Strands `toolChoice.any` has
no WebLLM equivalent and raises an error; auto and named choices are mapped.

Strands cache-point hints are ignored because they do not control WebLLM's KV
cache. Unsupported content, including S3 media, video, documents, guardrails,
reasoning blocks, and images inside tool results, raises an error rather than
being dropped. Multiple user text blocks are joined into one text field, followed
by image/audio parts, to fit WebLLM's one-text-part limit.

This is not a claim of 100% Strands integration for every WebLLM feature or model.
