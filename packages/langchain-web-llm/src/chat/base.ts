import { modelProfile } from './profile.js'
import type { ModelProfile } from '@langchain/core/language_models/profile'
import type { StructuredOutputMethodOptions } from '@langchain/core/language_models/base'
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
import type { Runnable } from '@langchain/core/runnables'
import type { BaseLanguageModelInput } from '@langchain/core/language_models/base'
import type {
  AIMessageChunk,
  BaseMessage,
  StandardMessageStructure,
} from '@langchain/core/messages'
import {
  BaseChatModel,
  type BindToolsInput,
  type LangSmithParams,
} from '@langchain/core/language_models/chat_models'
import type { CallbackManagerForLLMRun } from '@langchain/core/callbacks/manager'
import type { ChatModelStreamEvent } from '@langchain/core/language_models/event'
import { ChatGenerationChunk, type ChatResult } from '@langchain/core/outputs'
import { AsyncCaller, type AsyncCallerParams } from '@langchain/core/utils/async_caller'
import { formatTools } from './tools.js'
import { normalizeResponseFormat } from '../../../../shared/web-llm/response.js'
import { stringifyJSON } from '../../../../shared/web-llm/json.js'
import type { WebLLMModelOptions, WebLLMCallOptions, WebLLMRequest } from '../types.js'
import { readEvents } from './stream.js'

export abstract class WebLLMChatModel extends BaseChatModel<
  WebLLMCallOptions,
  AIMessageChunk<StandardMessageStructure>
> {
  readonly modelId: string
  private readonly modelProfile: ModelProfile
  protected readonly retries: number
  private readonly callerParams: AsyncCallerParams
  protected readonly request: WebLLMRequest

  constructor({
    modelId,
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
    this.modelId = modelId
    this.retries = params.maxRetries ?? 0
    this.callerParams = {
      maxRetries: this.retries,
      ...(params.maxConcurrency !== undefined && { maxConcurrency: params.maxConcurrency }),
      ...(params.onFailedAttempt !== undefined && { onFailedAttempt: params.onFailedAttempt }),
    }
    this.modelProfile = modelProfile(modelId, profile)
    this.request = structuredClone({
      ...request,
      ...(temperature !== undefined && { temperature }),
      ...(maxTokens !== undefined && { max_tokens: maxTokens }),
      ...(topP !== undefined && { top_p: topP }),
      ...(stop !== undefined && { stop }),
    })
  }

  override get profile(): ModelProfile {
    return structuredClone(this.modelProfile)
  }

  protected createCaller(): AsyncCaller {
    return new AsyncCaller(this.callerParams)
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

  override invocationParams(options?: this['ParsedCallOptions']) {
    return { model: this.modelId, ...this.formatRequest(options) }
  }

  protected formatRequest(options?: this['ParsedCallOptions']): WebLLMRequest {
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
    if (config?.strict) throw new Error('WebLLM cannot guarantee strict tool schemas.')
    const json =
      isInteropZodSchema(schema) || isSerializableSchema(schema) ? toJsonSchema(schema) : schema
    const method =
      config?.method ?? (this.profile.toolCalling === false ? 'jsonSchema' : 'functionCalling')
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
