import { Message, type ModelStreamEvent } from '@strands-agents/sdk'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { WebLLMModel } from '../model.js'
import type { WebLLMChunk } from '../types.js'
import { toBrowserError } from './errors.js'
import type { BrowserBridge, BrowserRequest, BrowserResult } from './types.js'
import { browserStream, type BrowserPacket } from '../../../../shared/browser/stream.js'
import { browserCalls, type BrowserCall } from '../../../../shared/browser/calls.js'
import { markChunkCallbackNonMutating } from '../../../../shared/web-llm/callback.js'

/** Serve a loaded engine on a private browser page. The caller owns its lifetime. */
export function createBrowserBridge(engine: MLCEngineInterface): BrowserBridge {
  type Read = BrowserPacket<ModelStreamEvent, WebLLMChunk>
  interface Call extends BrowserCall<Read> {
    takeChunks?: () => WebLLMChunk[]
  }
  const active = browserCalls<Read, Call>(engine)
  const { calls } = active
  const bridge: BrowserBridge = {
    withEngine: active.withEngine,
    async start({ messages, config, options, includeChunks }: BrowserRequest) {
      active.checkOpen()
      const id = crypto.randomUUID()
      const controller = new AbortController()
      const output = includeChunks ? browserStream<ModelStreamEvent, WebLLMChunk>() : undefined
      const model = new WebLLMModel({
        engine,
        ...config,
        ...(output && { onChunk: markChunkCallbackNonMutating(output.push) }),
      })
      const stream = model.stream(messages.map(Message.fromJSON), {
        ...options,
        cancelSignal: controller.signal,
      })
      calls.set(id, {
        stream,
        controller,
        ...(output && { takeChunks: output.takeChunks }),
        next: output ? () => output.next(stream) : () => stream.next(),
      })
      return id
    },
    async next(id: string): Promise<BrowserResult> {
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
          ...(chunks?.length && { chunks }),
          error: toBrowserError(error),
          get errorJSON(): string {
            return JSON.stringify(this.error)
          },
        }
        return packet
      }
    },
    stop: active.stop,
    close: () => active.close((id) => bridge.stop(id)),
  }
  return bridge
}
