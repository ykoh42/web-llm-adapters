import {
  AIMessageChunk,
  coerceMessageLikeToMessage,
  type StandardMessageStructure,
} from '@langchain/core/messages'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { withEngine } from '../../../../shared/web-llm/engine/access.js'
import type { WebLLMChunk } from '../types.js'
import { WebLLMEmbeddings } from '../embeddings/model.js'
import { ChatWebLLM } from '../model.js'
import type { BrowserBridge, BrowserChunk } from './types.js'
import { browserStream, type BrowserPacket } from '../../../../shared/browser/stream.js'
import { failedAttempts, type FailedAttempt } from './failed-attempts.js'
import { serializeBrowserError, type BrowserError } from './errors.js'

/** Serve a caller-owned engine on a private browser page. Closing does not unload it. */
export function createBrowserBridge(engine: MLCEngineInterface): BrowserBridge {
  type Read = BrowserPacket<BrowserChunk, WebLLMChunk> | FailedAttempt
  interface Call {
    chunks: WebLLMChunk[]
    stream: AsyncGenerator<BrowserChunk>
    next: () => Promise<Read>
    pending?: Promise<Read>
    retry?: (error: BrowserError) => void
    controller: AbortController
    closing?: Promise<void>
  }
  const calls = new Map<string, Call>()
  let closed = false
  const nativeCalls = new Set<Promise<unknown>>()
  function checkOpen() {
    if (closed) throw new Error('The browser bridge is closed.')
  }
  const bridge: BrowserBridge = {
    async withEngine(run) {
      checkOpen()
      const call = withEngine(engine, (engine) => {
        checkOpen()
        return run(engine)
      })
      nativeCalls.add(call)
      try {
        return await call
      } finally {
        nativeCalls.delete(call)
      }
    },
    async embed({ config, texts, type, hostRetries }) {
      let call: Promise<number[][]> | undefined
      try {
        checkOpen()
        const model = new WebLLMEmbeddings({
          ...config,
          engine,
          ...(hostRetries && { maxRetries: 0, onFailedAttempt: () => {} }),
        })
        if (type === 'query' && texts.length !== 1)
          throw new Error('An embedding query needs one text.')
        call =
          type === 'query'
            ? model.embedQuery(texts[0]!).then((vector) => [vector])
            : model.embedDocuments(texts)
        nativeCalls.add(call)
        return await call
      } catch (error) {
        const packet = {
          error: serializeBrowserError(error),
          get errorJSON(): string {
            return JSON.stringify(this.error)
          },
        }
        return packet
      } finally {
        if (call) nativeCalls.delete(call)
      }
    },
    async start({ config, messages, options, includeChunks, includeFailedAttempts }) {
      checkOpen()
      const id = crypto.randomUUID()
      const controller = new AbortController()
      const chunks: WebLLMChunk[] = []
      const output = includeChunks ? browserStream<BrowserChunk, WebLLMChunk>(chunks) : undefined
      const failures = includeFailedAttempts
        ? failedAttempts<BrowserPacket<BrowserChunk, WebLLMChunk>>(controller.signal)
        : undefined
      const model = new ChatWebLLM({
        ...config,
        engine,
        ...(output && { onChunk: output.push }),
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
        chunks,
        next: failures ? () => failures.next(read) : read,
        ...(failures && { retry: failures.retry }),
      })
      return id
    },
    async next(id) {
      const call = calls.get(id)
      try {
        if (!call) throw new Error('Unknown browser call.')
        const pending = call.pending ?? call.next()
        // Reserve the next chunk before waiting, so overlapping reads cannot share it.
        call.pending = pending.then((result) => (result.done ? result : call.next()))
        void call.pending.catch(() => {})
        const result = await pending
        if (result.done) calls.delete(id)
        return result
      } catch (error) {
        calls.delete(id)
        const packet = {
          error: serializeBrowserError(error),
          get errorJSON(): string {
            return JSON.stringify(this.error)
          },
          ...(call?.chunks.length && { chunks: call.chunks.splice(0) }),
        }
        return packet
      }
    },
    retry(id, error) {
      calls.get(id)?.retry?.(error)
    },
    stop(id) {
      const call = calls.get(id)
      if (!call) return Promise.resolve()
      call.controller.abort()
      call.closing ??= (async () => {
        try {
          await call.stream.return(undefined)
        } finally {
          calls.delete(id)
        }
      })()
      return call.closing
    },
    async close() {
      closed = true
      const results = await Promise.allSettled([
        ...Array.from(calls.keys(), (id) => bridge.stop(id)),
        ...Array.from(nativeCalls, (call) => call.catch(() => {})),
      ])
      for (const result of results) if (result.status === 'rejected') throw result.reason
    },
  }
  return bridge
}
