# Pi AI WebLLM

Use a loaded [WebLLM](https://github.com/mlc-ai/web-llm) engine as a custom provider for Pi AI and Pi Agent Core, entirely inside the browser.

Not published yet. Run the [browser example](../../examples/pi-ai-web-llm/README.md) from this repository. After publication:

```sh
npm install pi-ai-web-llm @earendil-works/pi-ai @mlc-ai/web-llm
```

## Use

Create a worker that runs WebLLM:

```ts
// worker.ts
import { WebWorkerMLCEngineHandler } from '@mlc-ai/web-llm'

const handler = new WebWorkerMLCEngineHandler()
self.onmessage = (event: MessageEvent) => handler.onmessage(event)
```

Load the engine, then register the provider using Pi AI's public API:

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
    },
  )
  try {
    const models = createModels()
    models.setProvider(createWebLLMProvider({ engine, modelId, contextWindow }))
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

The first load fetches WebLLM's runtime and model files. Inference runs locally. A main-thread `MLCEngine` also works; the browser example uses a Worker by default.

You own the engine, model loading, and Worker. Creating or replacing a provider does not load or unload a model. Match `contextWindow` to the engine's actual loading configuration. To register multiple engines, assign distinct `providerId` values. Calls sharing the same engine run in order, including calls from the other adapters in this repository.

## Pi Agent Core

Register the provider as above, then pass the model collection's bound stream function:

```ts
import { Agent } from '@earendil-works/pi-agent-core'

const agent = new Agent({
  initialState: { model, thinkingLevel: 'low' },
  streamFn: models.streamSimple.bind(models),
})
await agent.prompt('What is 2 + 2?')
```

Install `@earendil-works/pi-agent-core` separately if you use this example. Pi executes and validates tools; the provider translates their definitions, calls, and results. The [runnable browser example](../../examples/pi-ai-web-llm/README.md) includes a complete `lookup` tool loop.

## Support

| Feature                                      | Behavior                                                                                        |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Text, thinking, tool calls                   | Pi `AssistantMessageEvent` streams; model-dependent                                             |
| Tool results and history                     | Pi 1.3 transcript system sections and tool updates are resolved before inference                |
| Cancellation                                 | `signal` interrupts generation without unloading; queued cancellation leaves active calls alone |
| Errors                                       | Terminal Pi `error` events retain received text and available usage                             |
| Token usage                                  | Native WebLLM counts; API cost is zero; cache and thinking breakdowns are not estimated         |
| Images                                       | Base64 Pi image blocks require `imageInputs: true` and a compatible loaded artifact             |
| Constrained tool schemas                     | `strict: 'prefer'` uses ordinary tools; `strict: 'require'` and grammar tools are rejected      |
| Deferred requests and thinking token budgets | Unsupported                                                                                     |

Tool calls are published after native generation finishes and all calls have valid JSON object arguments. Truncated or malformed calls never become executable tool events. Pi still validates arguments against the tool schema before execution. Missing or mismatched tool results in replayed history produce errors; the adapter does not invent results.

Images from tool results are sent as a user turn after the tool results, matching WebLLM's input format. Interleaved text around images cannot be preserved and is rejected. Images are not part of the verified model matrix below.

## Thinking and native options

`streamSimple` and `completeSimple` follow Pi's thinking controls: omitted `reasoning` disables thinking; any supported Pi reasoning level enables it. Levels do not select different native token budgets. `Agent` therefore respects `thinkingLevel: 'off'`. Models that cannot reason reject an explicit reasoning request. Set `reasoning` on the provider options to override the model-family capability default.

Full `stream` and `complete` calls preserve the native `request.extra_body.enable_thinking` default. Other native generation options can be set through `request` at construction or on full stream calls. Pi's `temperature`, `maxTokens`, and `samplingParams` also work; sampling parameters override named generation options. Request options cannot replace the provider's model, conversation, tools, or streaming mode.

`onPayload` can inspect or replace the prepared native payload while preserving the loaded model and one streaming response. `onProviderStreamEvent` observes unchanged native chunks and is awaited in order. The provider-level `onChunk` is a synchronous native chunk observer. Callback failures fail the request and release the engine. HTTP-only options such as `fetch`, headers, retries, and `onResponse` do not apply to local inference. `timeoutMs` cancels a queued or active request.

Qwen tools cannot be combined with native JSON output. Gemma 4 tool/thinking mode uses the bundled [runtime and protocol](../../shared/runtime/README.md) and accepts text and tool results only. Exact model artifacts determine support.

## Verified models

Real WebLLM Worker checks exercise Pi Agent Core → tool call → tool result → final answer, native usage, cancellation, and recovery:

| WebLLM model ID                   | Tools | Thinking |
| --------------------------------- | ----- | -------- |
| `Qwen3.5-4B-q4f16_1-MLC`          | Yes   | Yes      |
| `Qwen3-8B-q4f16_1-MLC`            | Yes   | Yes      |
| `gemma-4-E2B-it-q4f16_1-MLC`      | Yes   | Yes      |
| `Qwen2.5-3B-Instruct-q4f16_1-MLC` | Yes   | No       |

Tested against `@earendil-works/pi-ai` and `@earendil-works/pi-agent-core` 1.3.0, and WebLLM 0.2.86. Model-selected arguments still require validation. Run `npm run test:pi` from the repository root to repeat the browser checks.
