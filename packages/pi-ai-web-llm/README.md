# Pi AI WebLLM

Use a loaded [WebLLM](https://github.com/mlc-ai/web-llm) engine as a custom provider for Pi AI, Pi Agent Core, and Pi Durable, entirely inside the browser.

Install the package, or run the [Agent Core and Durable browser examples](https://github.com/ykoh42/web-llm-adapters/blob/main/examples/pi-ai-web-llm/README.md) from this repository:

```sh
npm install pi-ai-web-llm
```

- Yarn: also install `@mlc-ai/web-llm` and `@earendil-works/pi-ai`. Check compatible versions with `yarn explain peer-requirements`.
- pnpm: if dependency scripts are blocked, run [`pnpm approve-builds`](https://pnpm.io/cli/approve-builds) to allow or deny them, then rerun `pnpm install`.
- CommonJS: use `require('pi-ai-web-llm')` on Node.js 22.19 or later. With Yarn PnP on Node 23.0–23.4, use `import()`.
- TypeScript: `moduleResolution: "bundler"` retains more WebLLM option types than `NodeNext`. WebLLM's declarations may require `skipLibCheck` and leave some types unchecked.

## Use

Create a worker that runs WebLLM:

```ts
// worker.ts
import { WebWorkerMLCEngineHandler } from '@mlc-ai/web-llm'

const handler = new WebWorkerMLCEngineHandler()
self.onmessage = (event: MessageEvent) => handler.onmessage(event)
```

Load Qwen3.5 with WebLLM 0.2.86, then register the provider using Pi AI's public API:

```ts
import { CreateWebWorkerMLCEngine } from '@mlc-ai/web-llm'
import { createModels } from '@earendil-works/pi-ai/models'
import { createWebLLMProvider } from 'pi-ai-web-llm'

const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
const contextWindow = 32768
const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' })
try {
  const engine = await CreateWebWorkerMLCEngine(
    worker,
    modelId,
    {},
    {
      context_window_size: contextWindow,
      conv_config: {
        role_empty_sep: '\n<think>\n',
        stop_token_ids: [248044, 248046],
      },
    },
  )
  try {
    const models = createModels()
    models.setProvider(
      createWebLLMProvider({
        engine,
        modelId,
        contextWindow,
        thinkingPrefilled: true,
        request: {
          temperature: 1,
          top_p: 0.95,
          presence_penalty: 1.5,
          repetition_penalty: 1,
          extra_body: { enable_thinking: true },
        },
      }),
    )
    const model = models.getModel('webllm', modelId)!
    const stream = models.streamSimple(
      model,
      {
        messages: [{ role: 'user', content: 'What is 2 + 2?', timestamp: Date.now() }],
      },
      { reasoning: 'low' },
    )
    for await (const event of stream) {
      if (event.type === 'text_delta') console.log(event.delta)
    }
    const result = await stream.result()
    if (result.stopReason === 'error' || result.stopReason === 'aborted') {
      throw new Error(result.errorMessage)
    }
  } finally {
    await engine.unload()
  }
} finally {
  worker.terminate()
}
```

The first load fetches WebLLM's runtime and model files. Inference runs locally. A main-thread `MLCEngine` also works; the browser example uses a Worker by default. For the following snippets, put imports at the top of the file and other code inside the inner `try` block.

While the engine is loaded, select a registered function for an API-specific call:

```ts
import { Type } from '@earendil-works/pi-ai'

const toolResult = await models.complete(
  model,
  {
    messages: [{ role: 'user', content: 'Look up A7.', timestamp: Date.now() }],
    tools: [
      {
        name: 'lookup',
        description: 'Look up a code.',
        parameters: Type.Object({ code: Type.String() }),
      },
    ],
  },
  { toolChoice: { type: 'function', function: { name: 'lookup' } } },
)
console.log(toolResult.content)
```

You own the engine, model loading, and Worker. Creating or replacing a provider does not load or unload a model. Match `contextWindow` to the engine's actual loading configuration. To register multiple engines, assign distinct `providerId` values. Calls sharing the same engine run in order, including calls from the other adapters in this repository.

For other native APIs, import `withEngine` from `pi-ai-web-llm` and await all work inside `withEngine(engine, callback)`, including consuming streams.

With WebLLM 0.2.86, set `request.extra_body.resumable.enabled` to `true` to checkpoint chat, then call `resumeChatCompletion()` through `withEngine()`. Set `resumable.strictPersistence` to `true` to surface storage errors. WebLLM 0.2.79–0.2.85 ignores this option. WebLLM only resumes text-only requests without structured output or a custom logit processor; this does not restore Pi transcripts or agent state. Gemma tool and thinking requests use raw completions and cannot resume.

## Pi Agent Core

While the engine is loaded, pass `models.streamSimple.bind(models)` to the agent:

```ts
import { Agent } from '@earendil-works/pi-agent-core'

const agent = new Agent({
  initialState: { model, thinkingLevel: 'low' },
  streamFn: models.streamSimple.bind(models),
})
await agent.prompt('What is 2 + 2?')
```

Install `@earendil-works/pi-agent-core` separately. Pi validates and executes tools; [agent.ts](https://github.com/ykoh42/web-llm-adapters/blob/main/examples/pi-ai-web-llm/browser/src/agent.ts) shows a complete tool loop.

## Pi Durable

While the engine is loaded, pass the registered `models` to Durable's Harness:

```ts
import { BACKGROUND_CONTEXT as context } from '@earendil-works/chord/context'
import { createRegistry, Harness, MemoryStorage } from '@earendil-works/pi-durable'

const harness = await Harness.open(
  new MemoryStorage(),
  { models, registry: createRegistry() },
  context,
)
try {
  const root = await harness.root(context, {
    agent: { model: { provider: model.provider, modelId: model.id } },
  })
  const submission = await root.submit({ type: 'input', content: 'What is 2 + 2?' }, context)
  await submission.wait(context)
} finally {
  await harness.close(context)
}
```

Install `@earendil-works/pi-durable` and `@earendil-works/chord` separately. `MemoryStorage` is temporary; the [Durable example](https://github.com/ykoh42/web-llm-adapters/blob/main/examples/pi-ai-web-llm/browser/src/durable-agent.ts) adds a tool, IndexedDB persistence, and reload recovery.

## Support

| Feature                                      | Behavior                                                                                                         |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Text, thinking, tool calls                   | Pi `AssistantMessageEvent` streams; model-dependent                                                              |
| Tool results and history                     | Transcript system sections and tool updates are resolved before inference                                        |
| Cancellation                                 | `signal` interrupts generation without unloading; queued cancellation leaves active calls alone                  |
| Errors                                       | Terminal Pi `error` events retain received text and available usage                                              |
| Token usage                                  | Native counts; missing usage and cost are 0; cache and thinking breakdowns are not estimated                     |
| Response details and log probabilities       | `responseId`, `responseModel`, `rawStopReason`; use `request.logprobs: true` and `onChunk` for log probabilities |
| Images                                       | Built-in Phi-3.5 Vision IDs enable base64 Pi image blocks by default; custom artifacts need `imageInputs: true`  |
| Constrained tool schemas                     | `strict: 'prefer'` uses ordinary tools; `strict: 'require'` and grammar tools are rejected                       |
| Deferred requests and thinking token budgets | Unsupported                                                                                                      |

When WebLLM reuses conversation state, input usage may count only new prompt tokens. This can make context estimates too low.
Pi Models may report cancellation during its setup as `error` before the provider runs.

Available tool-call fragments are streamed. `toolcall_end` is emitted after generation finishes and all calls have valid JSON object arguments. Pi validates arguments against the tool schema before execution. Missing or mismatched tool results in replayed history produce errors; the adapter does not invent results. Tool errors are sent with an `[ERROR]` prefix because WebLLM has no tool-result status field.

Images from tool results are sent as a user turn after the tool results, matching WebLLM's input format. Text on both sides of an image is rejected in user messages. User image input is verified with `Phi-3.5-vision-instruct-q4f16_1-MLC`.

## Thinking and native options

For Gemma 4, `request.extra_body.preserve_thinking: true` also keeps earlier tool-call reasoning; the default keeps only the current user turn.

Set `thinkingPrefilled: true` at construction if your loaded Qwen template already opens `<think>`. This only controls output parsing; it does not change the engine or prompt. See [Qwen3.5 engine setup](https://github.com/ykoh42/web-llm-adapters#qwen35-engine-setup) for WebLLM 0.2.86.

Qwen thinking and tool parsing require a recognized WebLLM model ID; `reasoning: true` alone does not enable them for a custom ID. Custom IDs starting with `Qwen3-` or `Qwen3.5-` default to `reasoning: true`; set `reasoning: false` to disable Pi's thinking controls.

`streamSimple` and `completeSimple` follow Pi's thinking controls: omitted `reasoning` selects `off`; Pi chooses a supported level when needed. Levels do not select different native token budgets. `Agent` respects `thinkingLevel: 'off'` when the model supports it. Models that cannot reason reject an explicit reasoning request. Set `reasoning` on the provider options to override the model-family capability default.

Full `stream` and `complete` calls preserve the native `request.extra_body.enable_thinking` default. Other native generation options can be set through `request` at construction or on full stream calls. A higher-priority `extra_body` replaces earlier WebLLM fields; `preserve_thinking` is retained unless overridden. Pi's `temperature` and `maxTokens` also work; `samplingParams` forwards WebLLM options and overrides named generation options, including thinking. Native `max_tokens: null` removes the output token cap. Request options cannot replace the provider's model, conversation, tools, or streaming mode.

`onPayload` can inspect or replace the prepared native payload while preserving the loaded model and one streaming response. When setting JSON mode there, include a schema such as `'{}'`. On WebLLM's native chat path, it may narrow Pi's registered tools but cannot add unknown tools or select a removed tool; returned calls are checked against that list. Set Qwen and Gemma tools and `toolChoice` before `onPayload`; set Gemma thinking beforehand too. Gemma requests must keep the open protocol prefix intact. `onProviderStreamEvent` and provider-level `onChunk` callbacks are awaited in order; do not await work on the same engine from them. Callback failures fail the request. HTTP-only options such as `fetch`, headers, retries, and `onResponse` do not apply to local inference. `timeoutMs` cancels a queued or active request.

Qwen tools cannot be combined with native JSON output. For optional [Qwen tool-call constraints](https://github.com/ykoh42/web-llm-adapters#tool-call-constraints), set `request.constrainedToolCalls: true` with WebLLM 0.2.86. Install WebLLM `0.2.79` or later in the `0.2.x` series. Gemma 4 E2B was verified with WebLLM `0.2.86` and needs the bundled [runtime and protocol](https://github.com/ykoh42/web-llm-adapters/blob/main/shared/runtime/README.md). Gemma tools and thinking accept text and tool results only. Exact model artifacts determine support.

## Verified models

See the [verified models](https://github.com/ykoh42/web-llm-adapters#models) for exact IDs, features, and verified adapters.

Tested with Pi AI and Pi Agent Core **1.0.2** and **1.4.0**, and WebLLM **0.2.79** and **0.2.86**. Model availability depends on the installed WebLLM version.

WebLLM 0.2.79 ignores request-level `repetition_penalty` and `enable_latency_breakdown`. Versions 0.2.81–0.2.83 fail when directly imported as ESM in Node.js. Versions 0.2.79–0.2.85 can leave the engine blocked after a stream stops early. Use 0.2.86 for cancellation and reuse.

Qwen3.5 preserves object/array arguments, type-less non-string `enum`/`const` values, and `anyOf`/`oneOf` unions without string branches as JSON. History rejects `<|im_start|>`, `<|im_end|>`, or `<|endoftext|>` in tool arguments, `>` in top-level argument names, and `<parameter=` / `</parameter>` in raw strings.
