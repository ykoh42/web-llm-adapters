import {
  AIMessageChunk,
  coerceMessageLikeToMessage,
  type StandardMessageStructure,
} from '@langchain/core/messages'
import { serializeError } from '../../../../shared/web-llm/errors.js'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import type { WebLLMChunk } from '../types.js'
import { WebLLMEmbeddings } from '../embeddings/model.js'
import { ChatWebLLM } from '../model.js'
import type { BrowserBridge, BrowserChunk } from './types.js'
import { browserStream, type BrowserPacket } from '../../../../shared/browser/stream.js'

/** Serve a caller-owned engine on a private browser page. Closing does not unload it. */
export function createBrowserBridge(engine: MLCEngineInterface): BrowserBridge {
  interface Call {
    chunks: WebLLMChunk[]
    stream: AsyncGenerator<BrowserChunk>
    next: () => Promise<BrowserPacket<BrowserChunk, WebLLMChunk>>
    pending?: Promise<BrowserPacket<BrowserChunk, WebLLMChunk>>
    controller: AbortController
    closing?: Promise<void>
  }
  const calls = new Map<string, Call>()
  let closed = false
  const nativeCalls = new Set<Promise<unknown>>()
  const bridge: BrowserBridge = {
    async embed({ config, texts, type }) {
      let call: Promise<number[][]> | undefined
      try {
        if (closed) throw new Error('The browser bridge is closed.')
        const model = new WebLLMEmbeddings({ ...config, engine })
        if (type === 'query' && texts.length !== 1)
          throw new Error('An embedding query needs one text.')
        call =
          type === 'query'
            ? model.embedQuery(texts[0]!).then((vector) => [vector])
            : model.embedDocuments(texts)
        nativeCalls.add(call)
        return await call
      } catch (error) {
        return { error: serializeError(error) }
      } finally {
        if (call) nativeCalls.delete(call)
      }
    },
    async start({ config, messages, options, includeChunks }) {
      if (closed) throw new Error('The browser bridge is closed.')
      const id = crypto.randomUUID()
      const controller = new AbortController()
      const chunks: WebLLMChunk[] = []
      const output = includeChunks ? browserStream<BrowserChunk, WebLLMChunk>(chunks) : undefined
      const model = new ChatWebLLM({
        ...config,
        engine,
        ...(output && { onChunk: output.push }),
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
      calls.set(id, {
        stream: source,
        controller,
        chunks,
        next: output ? () => output.next(source) : () => source.next(),
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
        return {
          error: serializeError(error),
          ...(call?.chunks.length && { chunks: call.chunks.splice(0) }),
        }
      }
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
