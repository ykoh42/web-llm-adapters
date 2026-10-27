import { createProvider } from '@earendil-works/pi-ai/models'
import type {
  Api,
  Model,
  Provider,
  TranscriptContext,
  ProviderStreams,
} from '@earendil-works/pi-ai'
import type { ChatCompletionRequestStreaming } from '@mlc-ai/web-llm'
import { protocolStream } from '../../../shared/web-llm/protocol.js'
import { requestOptions, positiveInteger } from './request.js'
import { streamHooks } from './callbacks.js'
import { formatContext } from './messages.js'
import { responseStream } from './events.js'
import type { ToolPolicy } from './tools.js'
import { checkThinkingPrefill } from '../../../shared/web-llm/qwen/prefill.js'
import { isVisionModel, protocolModelId } from '../../../shared/web-llm/models.js'
import type { ToolChoice } from '../../../shared/web-llm/types.js'
import type { WebLLMApi, WebLLMProviderOptions, WebLLMStreamOptions } from './types.js'

/** Wrap a loaded browser engine without taking ownership of its worker or GPU memory. */
export function createWebLLMProvider(options: WebLLMProviderOptions): Provider<WebLLMApi> {
  const { engine, modelId, sourceModelId, onChunk, thinkingPrefilled = false } = options
  if (!modelId.trim()) throw new Error('modelId must name a loaded WebLLM model.')
  const source = protocolModelId(modelId, sourceModelId)
  checkThinkingPrefill(source, thinkingPrefilled)
  positiveInteger(options.contextWindow, 'contextWindow')
  const maxTokens = options.maxTokens ?? Math.min(4096, options.contextWindow)
  positiveInteger(maxTokens, 'maxTokens')
  if (maxTokens > options.contextWindow) throw new Error('maxTokens cannot exceed contextWindow.')
  const providerId = options.providerId ?? 'webllm'
  if (!providerId.trim()) throw new Error('providerId cannot be empty.')
  const defaults = structuredClone(options.request ?? {})
  const imageInputs = options.imageInputs ?? isVisionModel(source)
  const model: Model<WebLLMApi> = {
    id: modelId,
    name: options.name ?? modelId,
    api: 'webllm',
    provider: providerId,
    baseUrl: '',
    input: imageInputs ? ['text', 'image'] : ['text'],
    reasoning: options.reasoning ?? /^(Qwen3(?:\.5)?-|gemma-4-)/i.test(source),
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
    const choice = protocolToolChoice(call.toolChoice)
    return responseStream(
      requestModel,
      (message) => {
        if (
          requestModel.id !== modelId ||
          requestModel.api !== 'webllm' ||
          requestModel.provider !== providerId
        )
          throw new Error('This provider only serves its configured WebLLM model.')
        const request = requestOptions(requestModel, defaults, call, simple, context)
        const { history, tools } = formatContext(context, requestModel)
        const toolPolicy: ToolPolicy = { choice, toolNames: tools.map((tool) => tool.name) }
        const chunks = protocolStream({
          engine,
          modelId,
          ...(sourceModelId !== undefined && { sourceModelId }),
          thinkingPrefilled,
          history,
          tools,
          choice,
          request,
          onChunkMayMutate: Boolean(onChunk),
          onPrepared: (prepared) => updateToolPolicy(prepared, tools, toolPolicy),
          onChunk: (chunk) => {
            message.responseId ??= chunk.id
            message.responseModel ??= chunk.model
            const reason = chunk.choices[0]?.finish_reason
            if (reason) message.rawStopReason = reason
            return onChunk?.(chunk)
          },
          signal,
          ...streamHooks(requestModel, modelId, call, signal),
        })
        return { chunks, toolPolicy }
      },
      signal,
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

function protocolToolChoice(
  choice: WebLLMStreamOptions['toolChoice'],
): ToolChoice | 'none' | undefined {
  if (choice === 'none') return 'none'
  if (!choice || typeof choice !== 'object' || choice.type !== 'function') return undefined
  const name = choice.function?.name
  return typeof name === 'string' ? { tool: { name } } : undefined
}

function updateToolPolicy(
  request: ChatCompletionRequestStreaming,
  registered: readonly { name: string }[],
  policy: ToolPolicy,
) {
  const toolNames = request.tools?.map((tool) => tool.function.name) ?? []
  const registeredNames = new Set(registered.map((tool) => tool.name))
  if (toolNames.some((name) => !name || !registeredNames.has(name)))
    throw new Error('onPayload cannot add tools that Pi has not registered.')
  if (new Set(toolNames).size !== toolNames.length)
    throw new Error('onPayload must keep tool names unique.')

  const choice = request.tool_choice
  const toolChoice: ToolChoice | 'none' | undefined =
    choice === 'none'
      ? 'none'
      : !choice || choice === 'auto'
        ? undefined
        : { tool: { name: choice.function.name } }
  if (
    toolChoice &&
    typeof toolChoice === 'object' &&
    'tool' in toolChoice &&
    !toolNames.includes(toolChoice.tool.name)
  )
    throw new Error('onPayload selected a tool that is not in its tools list.')

  policy.choice = toolChoice
  policy.toolNames = toolNames
}
