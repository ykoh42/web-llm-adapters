import {
  AIMessageChunk,
  coerceMessageLikeToMessage,
  type StandardMessageStructure,
} from '@langchain/core/messages'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import type { WebLLMChunk } from '../types.js'
import { WebLLMEmbeddings } from '../embeddings/model.js'
import { ChatWebLLM } from '../model.js'
import type { BrowserBridge, BrowserChunk } from './types.js'
import { browserStream, type BrowserPacket } from '../../../../shared/browser/stream.js'
import { browserCalls, type BrowserCall } from '../../../../shared/browser/calls.js'
import { markChunkCallbackNonMutating } from '../../../../shared/web-llm/callback.js'
import { failedAttempts, type FailedAttempt } from './failed-attempts.js'
import { serializeBrowserError, type BrowserError } from './errors.js'

/** Serve a caller-owned engine on a private browser page. Closing does not unload it. */
export function createBrowserBridge(engine: MLCEngineInterface): BrowserBridge {
  type Read = BrowserPacket<BrowserChunk, WebLLMChunk> | FailedAttempt
  interface Call extends BrowserCall<Read> {
    takeChunks?: () => WebLLMChunk[]
    retry?: (error: BrowserError) => void
  }
  const active = browserCalls<Read, Call>(engine)
  const { calls } = active
  const bridge: BrowserBridge = {
    withEngine: active.withEngine,
    async embed({ config, texts, type, hostRetries }) {
      try {
        active.checkOpen()
        const model = new WebLLMEmbeddings({
          ...config,
          engine,
          ...(hostRetries && { maxRetries: 0, onFailedAttempt: () => {} }),
        })
        if (type === 'query' && texts.length !== 1)
          throw new Error('An embedding query needs one text.')
        const call =
          type === 'query'
            ? model.embedQuery(texts[0]!).then((vector) => [vector])
            : model.embedDocuments(texts)
        return await active.track(call)
      } catch (error) {
        const packet = {
          error: serializeBrowserError(error),
          get errorJSON(): string {
            return JSON.stringify(this.error)
          },
        }
        return packet
      }
    },
    async start({ config, messages, options, includeChunks, includeFailedAttempts }) {
      active.checkOpen()
      const id = crypto.randomUUID()
      const controller = new AbortController()
      const output = includeChunks ? browserStream<BrowserChunk, WebLLMChunk>() : undefined
      const failures = includeFailedAttempts
        ? failedAttempts<BrowserPacket<BrowserChunk, WebLLMChunk>>(controller.signal)
        : undefined
      const model = new ChatWebLLM({
        ...config,
        engine,
        ...(output && { onChunk: markChunkCallbackNonMutating(output.push) }),
        ...(failures && { onFailedAttempt: failures.handle }),
      })
      async function* stream(): AsyncGenerator<BrowserChunk> {
        for await (const chunk of model._streamResponseChunks(
          messages.map(coerceMessageLikeToMessage),
          { ...options, signal: controller.signal },
        )) {
          if (!AIMessageChunk.isInstance(chunk.message))
            throw new Error('WebLLM returned a non-AI chunk.')
          const message = chunk.message as AIMessageChunk<StandardMessageStructure>
          yield {
            text: chunk.text,
            message: {
              content: message.content,
              ...(message.id && { id: message.id }),
              ...(message.tool_call_chunks && { tool_call_chunks: message.tool_call_chunks }),
              ...(message.tool_calls?.length && { tool_calls: message.tool_calls }),
              ...(message.usage_metadata && { usage_metadata: message.usage_metadata }),
              response_metadata: message.response_metadata,
            },
            ...(chunk.generationInfo && { generationInfo: chunk.generationInfo }),
          }
        }
      }
      const source = stream()
      const read = output ? () => output.next(source) : () => source.next()
      calls.set(id, {
        stream: source,
        controller,
        ...(output && { takeChunks: output.takeChunks }),
        next: failures ? () => failures.next(read) : read,
        ...(failures && { retry: failures.retry }),
      })
      return id
    },
    async next(id) {
      const call = calls.get(id)
      try {
        if (!call) throw new Error('Unknown browser call.')
        const result = await active.read(call)
        if (result.done) calls.delete(id)
        return result
      } catch (error) {
        calls.delete(id)
        const chunks = call?.takeChunks?.()
        const packet = {
          error: serializeBrowserError(error),
          get errorJSON(): string {
            return JSON.stringify(this.error)
          },
          ...(chunks?.length && { chunks }),
        }
        return packet
      }
    },
    retry(id, error) {
      calls.get(id)?.retry?.(error)
    },
    stop: active.stop,
    close: () => active.close((id) => bridge.stop(id)),
  }
  return bridge
}
