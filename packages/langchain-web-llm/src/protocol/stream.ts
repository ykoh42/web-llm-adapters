import type { AsyncCaller } from '@langchain/core/utils/async_caller'
import { qwenHistory, trimThoughts } from './history.js'
import type { ChatCompletionRequestStreaming, MLCEngineInterface } from '@mlc-ai/web-llm'
import { gemmaPrompt, type GemmaMessage } from '../../../../shared/web-llm/gemma/prompt.js'
import { gemmaChunks } from '../../../../shared/web-llm/gemma/stream.js'
import { qwenChunks } from '../../../../shared/web-llm/qwen/stream.js'
import { qwen35Chunks } from '../../../../shared/web-llm/qwen35/stream.js'
import { isQwen } from '../../../../shared/web-llm/qwen/models.js'
import { isQwen35 } from '../../../../shared/web-llm/qwen35/models.js'
import { streamCall } from '../../../../shared/web-llm/engine/call.js'
import { stopReason, type StreamChunk } from '../../../../shared/web-llm/stream/chunks.js'
import type { ToolSpec, ToolChoice } from '../../../../shared/web-llm/types.js'
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
}: ProtocolInput): AsyncIterable<StreamChunk> {
  const open = <T>(run: () => Promise<T>) =>
    caller.call(() => {
      signal?.throwIfAborted()
      return run()
    })
  const gemma = /^gemma-4-/i.test(modelId)
  const structured =
    request.response_format?.type !== undefined && request.response_format.type !== 'text'
  if (choice === 'none') tools = []
  const thinking = request.extra_body?.enable_thinking === true
  if (gemma) history = trimThoughts(history)
  const hasHistory = history.some(
    (message) =>
      message.role === 'tool' ||
      (message.role === 'assistant' && (message.tool_calls?.length || message.reasoning)),
  )
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
  if (gemma && (tools.length || thinking || hasHistory)) {
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
      (chunks) => gemmaChunks(chunks, tools, onChunk, prefix, Boolean(request.stop?.length)),
      signal,
    )
  }
  if (isQwen(modelId) || isQwen35(modelId)) {
    if (tools.length && structured)
      throw new Error('Qwen tool calls cannot use a structured response format.')
    if (
      modelId === 'Qwen3-1.7B-q4f16_1-MLC' &&
      (tools.length ||
        history.some(
          (message) =>
            message.role === 'tool' || (message.role === 'assistant' && message.tool_calls?.length),
        ) ||
        (choice && choice !== 'none'))
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
          ? qwen35Chunks(chunks, tools, onChunk, !structured)
          : qwenChunks(chunks, tools, onChunk, !structured),
      signal,
    )
  }
  const native: ChatCompletionRequestStreaming = { ...sampling, messages: nativeMessages(history) }
  if (tools.length)
    native.tools = tools.map((tool) => ({
      type: 'function',
      function: {
        name: tool.name,
        description: tool.description,
        parameters: { ...(tool.inputSchema ?? { type: 'object', properties: {} }) },
      },
    }))
  if (choice === 'none') native.tool_choice = 'none'
  else if (choice && 'any' in choice)
    throw new Error('WebLLM does not support required tool choice for this model.')
  else if (choice && 'tool' in choice)
    native.tool_choice = { type: 'function', function: { name: choice.tool.name } }
  return streamCall(
    engine,
    () => open(() => engine.chat.completions.create(native)),
    async function* (chunks) {
      for await (const chunk of chunks) {
        onChunk?.(chunk)
        const first = chunk.choices[0]
        if (chunk.choices.length > 1 || (first && first.index !== 0))
          throw new Error('WebLLM adapters need one response per request.')
        yield {
          text: tools.length ? '' : (first?.delta.content ?? ''),
          reasoning: '',
          toolText: tools.length ? (first?.delta.content ?? '') : '',
          calls: (first?.delta.tool_calls ?? []).map((call) => ({
            index: call.index,
            id: call.id,
            name: call.function?.name,
            input: call.function?.arguments,
          })),
          stop: stopReason(first?.finish_reason),
          usage: chunk.usage,
        }
      }
    },
    signal,
  )
}
