import { modelProfile } from './profile.js'
import type { ModelProfile } from '@langchain/core/language_models/profile'
import type {
  BaseLanguageModelInput,
  StructuredOutputMethodOptions,
} from '@langchain/core/language_models/base'
import {
  assembleStructuredOutputPipeline,
  createContentParser,
  createFunctionCallingParser,
} from '@langchain/core/language_models/structured_output'
import { toJsonSchema } from '@langchain/core/utils/json_schema'
import { isInteropZodSchema, type InteropZodType } from '@langchain/core/utils/types'
import {
  isSerializableSchema,
  type SerializableSchema,
} from '@langchain/core/utils/standard_schema'
import type { Runnable, RunnableConfig } from '@langchain/core/runnables'
import {
  AIMessageChunk,
  type BaseMessage,
  type StandardMessageStructure,
} from '@langchain/core/messages'
import {
  BaseChatModel,
  type BindToolsInput,
  type LangSmithParams,
} from '@langchain/core/language_models/chat_models'
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager'
import type { ChatModelStreamEvent } from '@langchain/core/language_models/event'
import { ChatGenerationChunk, type ChatResult } from '@langchain/core/outputs'
import type { AsyncCaller, AsyncCallerParams } from '@langchain/core/utils/async_caller'
import { formatTools } from './tools.js'
import { normalizeResponseFormat } from '../../../../shared/web-llm/response.js'
import { stringifyJSON } from '../../../../shared/web-llm/json.js'
import type { WebLLMModelOptions, WebLLMCallOptions, WebLLMRequest } from '../types.js'
import { readEvents } from './events.js'
import { SignalCaller } from './caller.js'
import { checkRequest, preserveThinkingRequest } from '../../../../shared/web-llm/request.js'
import { checkThinkingPrefill } from '../../../../shared/web-llm/qwen/prefill.js'

export abstract class WebLLMChatModel extends BaseChatModel<
  WebLLMCallOptions,
  AIMessageChunk<StandardMessageStructure>
> {
  readonly modelId: string
  private readonly modelProfile: ModelProfile
  protected readonly retries: number
  private readonly callerParams: AsyncCallerParams
  protected readonly request: WebLLMRequest
  protected readonly thinkingPrefilled: boolean

  constructor({
    modelId,
    thinkingPrefilled = false,
    request,
    temperature,
    maxTokens,
    topP,
    stop,
    profile,
    ...params
  }: WebLLMModelOptions) {
    super({ ...params, maxRetries: params.maxRetries ?? 0 })
    if (!modelId) throw new Error('modelId must name a loaded WebLLM model.')
    checkThinkingPrefill(modelId, thinkingPrefilled)
    this.thinkingPrefilled = thinkingPrefilled
    this.disableStreaming = params.disableStreaming ?? false
    this.modelId = modelId
    this.retries = params.maxRetries ?? 0
    this.callerParams = {
      maxRetries: this.retries,
      ...(params.maxConcurrency !== undefined && { maxConcurrency: params.maxConcurrency }),
      ...(params.onFailedAttempt !== undefined && { onFailedAttempt: params.onFailedAttempt }),
    }
    this.modelProfile = modelProfile(modelId, profile, request)
    this.request = structuredClone({
      ...request,
      ...(temperature !== undefined && { temperature }),
      ...(maxTokens !== undefined && { max_tokens: maxTokens }),
      ...(topP !== undefined && { top_p: topP }),
      ...(stop !== undefined && { stop }),
    })
    checkRequest(this.request)
  }

  override get profile(): ModelProfile {
    return structuredClone(this.modelProfile)
  }

  protected createCaller(signal: AbortSignal): AsyncCaller {
    return new SignalCaller(this.callerParams, signal)
  }

  static override lc_name() {
    return 'ChatWebLLM'
  }
  override _llmType() {
    return 'webllm'
  }
  override get callKeys() {
    return [
      ...super.callKeys,
      'tools',
      'request',
      'response_format',
      'tool_choice',
      'ls_structured_output_format',
    ]
  }
  override _identifyingParams() {
    return this.invocationParams()
  }

  // Cache the effective output format and tool definitions, not tool instances.
  override _getSerializedCacheKeyParametersForCall(
    options: WebLLMCallOptions & { config?: RunnableConfig },
  ): string {
    return super._getSerializedCacheKeyParametersForCall({
      ...options,
      ...(options.tools && { tools: formatTools(options.tools) }),
      outputVersion: options.outputVersion ?? this.outputVersion ?? 'v0',
      ...(this.thinkingPrefilled && { thinkingPrefilled: true }),
    })
  }

  // Keep inferred fields available with older WebLLM peer types.
  override invocationParams(options?: this['ParsedCallOptions']) {
    return { model: this.modelId, ...this.formatRequest(options) }
  }

  protected formatRequest(options?: this['ParsedCallOptions']): WebLLMRequest {
    if (options?.request) checkRequest(options.request)
    const format = options?.response_format
    const request = structuredClone({
      ...this.request,
      ...options?.request,
      ...(options?.stop && { stop: options.stop }),
      ...(format && {
        response_format:
          format.type === 'json_schema'
            ? { type: 'json_object' as const, schema: stringifyJSON(format.json_schema.schema) }
            : format,
      }),
    })
    preserveThinkingRequest(request, options?.request?.extra_body, this.request.extra_body)
    if (request.response_format)
      request.response_format = normalizeResponseFormat(request.response_format)
    return request
  }

  override getLsParams(options: this['ParsedCallOptions']): LangSmithParams {
    const request = this.invocationParams(options)
    return {
      ls_provider: 'webllm',
      ls_model_name: this.modelId,
      ls_model_type: 'chat',
      ...(request.temperature != null && { ls_temperature: request.temperature }),
      ...(request.max_tokens != null && { ls_max_tokens: request.max_tokens }),
      ...(request.stop != null && {
        ls_stop: typeof request.stop === 'string' ? [request.stop] : request.stop,
      }),
    }
  }

  override bindTools(
    tools: BindToolsInput[],
    options?: Partial<WebLLMCallOptions>,
  ): Runnable<BaseLanguageModelInput, AIMessageChunk<StandardMessageStructure>, WebLLMCallOptions> {
    // Validate before creating a runnable, while keeping the caller's tools untouched.
    formatTools(tools)
    return this.withConfig({ ...options, tools })
  }

  override withStructuredOutput<Output extends object = Record<string, unknown>>(
    schema: InteropZodType<Output> | SerializableSchema<Output> | Record<string, unknown>,
    config?: StructuredOutputMethodOptions<false>,
  ): Runnable<BaseLanguageModelInput, Output, WebLLMCallOptions>
  override withStructuredOutput<Output extends object = Record<string, unknown>>(
    schema: InteropZodType<Output> | SerializableSchema<Output> | Record<string, unknown>,
    config?: StructuredOutputMethodOptions<true>,
  ): Runnable<BaseLanguageModelInput, { raw: BaseMessage; parsed: Output }, WebLLMCallOptions>
  override withStructuredOutput<Output extends object = Record<string, unknown>>(
    schema: InteropZodType<Output> | SerializableSchema<Output> | Record<string, unknown>,
    config?: StructuredOutputMethodOptions<boolean>,
  ):
    | Runnable<BaseLanguageModelInput, Output, WebLLMCallOptions>
    | Runnable<BaseLanguageModelInput, { raw: BaseMessage; parsed: Output }, WebLLMCallOptions> {
    const json =
      isInteropZodSchema(schema) || isSerializableSchema(schema) ? toJsonSchema(schema) : schema
    const method =
      config?.method ?? (this.profile.toolCalling === false ? 'jsonSchema' : 'functionCalling')
    if (config?.strict && method !== 'jsonSchema')
      throw new Error('WebLLM strict mode requires jsonSchema.')
    const tracing = { kwargs: { method }, schema: json }
    if (method === 'jsonMode' || method === 'jsonSchema') {
      const llm = this.withConfig({
        response_format: {
          type: 'json_object',
          ...(method === 'jsonSchema' && { schema: stringifyJSON(json) }),
        },
        tool_choice: 'none',
        ls_structured_output_format: tracing,
      })
      return assembleStructuredOutputPipeline<Output>(
        llm,
        createContentParser(schema),
        config?.includeRaw,
      )
    }
    if (method !== 'functionCalling')
      throw new Error('Use functionCalling, jsonMode, or jsonSchema.')
    const name =
      config?.name ?? ('name' in json && typeof json.name === 'string' ? json.name : 'extract')
    const llm = this.bindTools(
      [
        {
          type: 'function',
          function: {
            name,
            description: typeof json.description === 'string' ? json.description : '',
            parameters: json,
          },
        },
      ],
      { tool_choice: name, ls_structured_output_format: tracing },
    )
    return assembleStructuredOutputPipeline<Output>(
      llm,
      createFunctionCallingParser(schema, name),
      config?.includeRaw,
    )
  }

  override async _generate(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun,
  ): Promise<ChatResult> {
    let result: ChatGenerationChunk | undefined
    for await (const chunk of this._streamResponseChunks(messages, options, runManager))
      result = result ? result.concat(chunk) : chunk
    if (!result) throw new Error('WebLLM returned no response.')
    return this.finishResult(result)
  }

  protected finishResult(result: ChatGenerationChunk): ChatResult {
    if (
      result.generationInfo?.finish_reason === 'length' &&
      AIMessageChunk.isInstance(result.message)
    ) {
      result.message.tool_calls = []
      result.message.invalid_tool_calls = []
      result.message.tool_call_chunks = []
    }
    return { generations: [result] }
  }

  override async *_streamChatModelEvents(
    messages: BaseMessage[],
    options: this['ParsedCallOptions'],
    runManager?: CallbackManagerForLLMRun,
  ): AsyncGenerator<ChatModelStreamEvent> {
    yield* readEvents(this._streamResponseChunks(messages, options, runManager), options.signal)
  }
}
