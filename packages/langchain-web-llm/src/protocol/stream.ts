import type { AsyncCaller } from '@langchain/core/utils/async_caller'
import { trimThoughts, qwenHistory } from './history.js'
import type { ChatCompletionRequestStreaming, MLCEngineInterface } from '@mlc-ai/web-llm'
import { gemmaPrompt, type GemmaMessage } from '../../../../shared/web-llm/gemma/prompt.js'
import { gemmaChunks } from '../../../../shared/web-llm/gemma/stream.js'
import { qwenChunks } from '../../../../shared/web-llm/qwen/stream.js'
import { qwen35Chunks } from '../../../../shared/web-llm/qwen35/stream.js'
import { isQwen } from '../../../../shared/web-llm/qwen/models.js'
import { isQwen35 } from '../../../../shared/web-llm/qwen35/models.js'
import { isTextFormat } from '../../../../shared/web-llm/response.js'
import { streamCall } from '../../../../shared/web-llm/engine/call.js'
import type { StreamChunk } from '../../../../shared/web-llm/stream/chunks.js'
import { nativeChunks, nativeHistory, nativeTools } from '../../../../shared/web-llm/native.js'
import type { ToolSpec, ToolChoice } from '../../../../shared/web-llm/types.js'
import { modelError } from '../error.js'
import { nativeMessages } from '../chat/messages.js'
import type { ChatWebLLMOptions, WebLLMRequest } from '../types.js'

interface ProtocolInput {
  caller: AsyncCaller
  engine: MLCEngineInterface
  modelId: string
  history: GemmaMessage[]
  tools: ToolSpec[]
  choice: ToolChoice | 'none' | undefined
  request: WebLLMRequest
  onChunk: ChatWebLLMOptions['onChunk']
  signal: AbortSignal | undefined
  maxRetries: number | undefined
}

export function protocolStream({
  engine,
  caller,
  modelId,
  history,
  tools,
  choice,
  request,
  onChunk,
  signal,
  maxRetries,
}: ProtocolInput): AsyncIterable<StreamChunk> {
  const open = <T>(run: () => Promise<AsyncIterable<T>>) => {
    let attempt: Promise<AsyncIterable<T>> | undefined
    const pending = caller.callWithOptions(
      {
        ...(maxRetries !== undefined && { maxRetries }),
        ...(signal && { signal }),
      },
      async () => {
        signal?.throwIfAborted()
        try {
          attempt = run()
          return await attempt
        } catch (error) {
          throw modelError(error)
        }
      },
    )
    if (!signal) return pending
    return pending.catch(async (error) => {
      if (signal.aborted && attempt) await closeUnclaimedStream(attempt).catch(() => {})
      throw error
    })
  }
  const gemma = /^gemma-4-/i.test(modelId)
  const structured = !isTextFormat(request.response_format)
  if (choice === 'none') tools = []
  const thinking = request.extra_body?.enable_thinking === true
  if (gemma) history = trimThoughts(history)
  const gemmaProtocol =
    gemma &&
    (tools.length > 0 ||
      thinking ||
      history.some(
        (message) =>
          message.role === 'tool' ||
          (message.role === 'assistant' && (message.tool_calls?.length || message.reasoning)),
      ))
  const sampling = {
    ...request,
    model: modelId,
    stream: true as const,
    stream_options: { include_usage: true },
    n: 1,
  }
  if (gemma && request.extra_body?.enable_thinking === false) {
    sampling.extra_body = { ...sampling.extra_body }
    delete sampling.extra_body.enable_thinking
  }
  if (gemmaProtocol) {
    if (structured || request.extra_body?.resumable?.enabled || request.ignore_eos)
      throw new Error(
        'Gemma protocol mode does not support constrained response_format, resume, or ignore_eos.',
      )
    const { extra_body, ...raw } = sampling
    delete raw.response_format
    let prompt = gemmaPrompt(history, tools, thinking)
    let prefix = ''
    if (choice && choice !== 'none' && 'any' in choice)
      throw new Error('Gemma does not support required tool choice. Choose auto or a named tool.')
    if (choice && choice !== 'none' && 'tool' in choice) {
      if (!tools.some((tool) => tool.name === choice.tool.name))
        throw new Error('The chosen Gemma tool is not registered.')
      prefix = '<|tool_call>call:' + choice.tool.name + '{'
    }
    if (thinking && !prefix && prompt.endsWith('<tool_response|>')) prefix = '<|channel>thought\n'
    prompt += prefix
    return streamCall(
      engine,
      () =>
        open(() =>
          engine.completions.create({
            ...raw,
            prompt,
            ...(extra_body?.enable_latency_breakdown !== undefined && {
              extra_body: { enable_latency_breakdown: extra_body.enable_latency_breakdown },
            }),
          }),
        ),
      (chunks) =>
        gemmaChunks(chunks, tools, {
          onChunk,
          prefix,
          stopRequested: Boolean(request.stop?.length),
          signal,
        }),
      signal,
    )
  }
  if (isQwen(modelId) || isQwen35(modelId)) {
    if (tools.length && structured)
      throw new Error(
        'Qwen tool calls cannot use a structured response format. For agents, use responseFormat: toolStrategy(schema).',
      )
    if (
      modelId === 'Qwen3-1.7B-q4f16_1-MLC' &&
      (tools.length ||
        history.some(
          (message) =>
            message.role === 'tool' || (message.role === 'assistant' && message.tool_calls?.length),
        ) ||
        (choice && choice !== 'none' && !('auto' in choice)))
    )
      throw new Error('Tool calling is not supported for Qwen3-1.7B-q4f16_1-MLC.')
    const xml = isQwen35(modelId)
    const messages = qwenHistory(
      modelId,
      history,
      tools,
      choice === 'none' ? undefined : choice,
      xml,
    )
    return streamCall(
      engine,
      () => open(() => engine.chat.completions.create({ ...sampling, messages })),
      (chunks) =>
        xml
          ? qwen35Chunks(chunks, tools, { onChunk, parseProtocol: !structured, signal })
          : qwenChunks(chunks, tools, {
              onChunk,
              parseProtocol: !structured,
              parseThinking: modelId.startsWith('Qwen3-'),
              signal,
            }),
      signal,
    )
  }
  if (
    choice !== 'none' &&
    choice &&
    'tool' in choice &&
    !tools.some((tool) => tool.name === choice.tool.name)
  )
    throw new Error('The chosen WebLLM tool is not registered.')
  const native: ChatCompletionRequestStreaming = {
    ...sampling,
    messages: nativeHistory(nativeMessages(history), modelId, Boolean(tools.length)),
  }
  if (tools.length) native.tools = nativeTools(tools)
  if (choice === 'none') native.tool_choice = 'none'
  else if (choice && 'any' in choice)
    throw new Error('WebLLM does not support required tool choice for this model.')
  else if (choice && 'tool' in choice)
    native.tool_choice = { type: 'function', function: { name: choice.tool.name } }
  return streamCall(
    engine,
    // Hermes adds its system prompt and response format while opening a request.
    () => open(() => engine.chat.completions.create({ ...native, messages: [...native.messages] })),
    (chunks) => nativeChunks(chunks, Boolean(native.tools), { onChunk, signal }),
    signal,
  )
}

async function closeUnclaimedStream<T>(pending: Promise<AsyncIterable<T>>) {
  let stream: AsyncIterable<T>
  try {
    stream = await pending
  } catch {
    return
  }

  await stream[Symbol.asyncIterator]().return?.()
}
