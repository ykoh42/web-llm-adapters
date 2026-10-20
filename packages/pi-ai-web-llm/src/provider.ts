import { createProvider } from '@earendil-works/pi-ai/models'
import type {
  Api,
  Model,
  Provider,
  TranscriptContext,
  ProviderStreams,
} from '@earendil-works/pi-ai'
import { protocolStream } from '../../../shared/web-llm/protocol.js'
import { requestOptions, positiveInteger } from './request.js'
import { streamHooks } from './callbacks.js'
import { formatContext } from './messages.js'
import { responseStream } from './events.js'
import type { WebLLMApi, WebLLMProviderOptions, WebLLMStreamOptions } from './types.js'

/** Wrap a loaded browser engine without taking ownership of its worker or GPU memory. */
export function createWebLLMProvider(options: WebLLMProviderOptions): Provider<WebLLMApi> {
  if (!options.modelId.trim()) throw new Error('modelId must name a loaded WebLLM model.')
  positiveInteger(options.contextWindow, 'contextWindow')
  const maxTokens = options.maxTokens ?? Math.min(4096, options.contextWindow)
  positiveInteger(maxTokens, 'maxTokens')
  if (maxTokens > options.contextWindow) throw new Error('maxTokens cannot exceed contextWindow.')
  const providerId = options.providerId ?? 'webllm'
  if (!providerId.trim()) throw new Error('providerId cannot be empty.')
  const defaults = structuredClone(options.request ?? {})
  const model: Model<WebLLMApi> = {
    id: options.modelId,
    name: options.name ?? options.modelId,
    api: 'webllm',
    provider: providerId,
    baseUrl: '',
    input: options.imageInputs ? ['text', 'image'] : ['text'],
    reasoning: options.reasoning ?? /^(Qwen3(?:\.5)?-|gemma-4-)/i.test(options.modelId),
    contextWindow: options.contextWindow,
    maxTokens,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  }

  function stream(
    requestModel: Model<Api>,
    context: TranscriptContext,
    call: WebLLMStreamOptions = {},
    simple = false,
  ) {
    const signal =
      call.timeoutMs === undefined
        ? call.signal
        : AbortSignal.any([
            ...(call.signal ? [call.signal] : []),
            AbortSignal.timeout(call.timeoutMs),
          ])
    const choice = call.toolChoice === 'none' ? 'none' : undefined
    return responseStream(
      requestModel,
      () => {
        if (
          requestModel.id !== model.id ||
          requestModel.api !== model.api ||
          requestModel.provider !== providerId
        )
          throw new Error('This provider only serves its configured WebLLM model.')
        const request = requestOptions(requestModel, defaults, call, simple)
        const { history, tools } = formatContext(context, requestModel)
        const chunks = protocolStream({
          engine: options.engine,
          modelId: options.modelId,
          history,
          tools,
          choice,
          request,
          onChunk: options.onChunk,
          signal,
          ...streamHooks(requestModel, options.modelId, call, signal),
        })
        return { chunks, toolNames: tools.map((tool) => tool.name) }
      },
      signal,
      choice,
    )
  }

  const api: ProviderStreams = {
    stream,
    streamSimple: (model, context, options) => stream(model, context, options, true),
  }
  return createProvider<WebLLMApi>({
    id: providerId,
    name: 'WebLLM',
    models: [model],
    api,
    auth: { apiKey: { name: 'WebLLM (local)', resolve: async () => ({ auth: {} }) } },
  })
}
