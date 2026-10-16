import {
  ModelError,
  type Message,
  type StreamOptions,
  type ModelStreamEvent,
} from '@strands-agents/sdk'
import type { MLCEngineInterface, CompletionCreateParamsStreaming } from '@mlc-ai/web-llm'
import type { WebLLMModelConfig, WebLLMChunk } from '../types.js'
import { startCall } from '../call.js'
import { readStream } from '../stream.js'
import { toolPrompt } from './prompt.js'
import { gemmaChunks } from './stream.js'

/** Gemma's native thinking and tool protocol through raw completions. */
export async function* gemmaStream(
  engine: MLCEngineInterface,
  config: WebLLMModelConfig,
  messages: Message[],
  options: StreamOptions,
  onChunk?: (chunk: WebLLMChunk) => void,
): AsyncGenerator<ModelStreamEvent> {
  const { response_format, extra_body, ...sampling } = config.request ?? {}
  if (response_format != null || extra_body?.resumable || sampling.ignore_eos) {
    throw new ModelError(
      'Gemma protocol mode does not support response_format, resume, or ignore_eos.',
    )
  }
  const thinking = extra_body?.enable_thinking === true
  const tools = options.toolSpecs ?? []
  let prefix = ''
  if (options.toolChoice) {
    if ('any' in options.toolChoice)
      throw new ModelError('WebLLM does not support toolChoice.any. Choose auto or a named tool.')
    if ('tool' in options.toolChoice) {
      const name = options.toolChoice.tool.name
      if (!tools.some((tool) => tool.name === name))
        throw new ModelError('The chosen Gemma tool is not registered.')
      prefix = '<|tool_call>call:' + name
    }
  }
  const prompt = toolPrompt(messages, tools, options, thinking)
  if (thinking && !prefix && prompt.endsWith('<tool_response|>')) prefix = '<|channel>thought\n'
  const request: CompletionCreateParamsStreaming = {
    ...sampling,
    model: config.modelId,
    prompt: prompt + prefix,
    stream: true,
    stream_options: { include_usage: true },
    n: 1,
    ...(extra_body?.enable_latency_breakdown !== undefined && {
      extra_body: { enable_latency_breakdown: extra_body.enable_latency_breakdown },
    }),
  }
  if (config.maxTokens !== undefined) request.max_tokens = config.maxTokens
  if (config.temperature !== undefined) request.temperature = config.temperature
  if (config.topP !== undefined) request.top_p = config.topP
  const call = await startCall(
    engine,
    () => engine.completions.create(request),
    options.cancelSignal,
  )
  let done = false
  try {
    const chunks = gemmaChunks(call.chunks, tools, onChunk, prefix)
    yield* readStream(chunks, false, options.cancelSignal)
    done = true
  } finally {
    await call.close(!done)
  }
}
