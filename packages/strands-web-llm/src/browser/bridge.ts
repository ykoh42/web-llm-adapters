import { Message, type ModelStreamEvent } from '@strands-agents/sdk'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { WebLLMModel } from '../model.js'
import { withEngine } from '../../../../shared/web-llm/engine/access.js'
import type { WebLLMChunk } from '../types.js'
import { toBrowserError } from './errors.js'
import type { BrowserBridge, BrowserRequest, BrowserResult } from './types.js'
import { browserStream, type BrowserPacket } from '../../../../shared/browser/stream.js'
import { markChunkCallbackNonMutating } from '../../../../shared/web-llm/callback.js'

/** Serve a loaded engine on a private browser page. The caller owns its lifetime. */
export function createBrowserBridge(engine: MLCEngineInterface): BrowserBridge {
  interface Call {
    stream: ReturnType<WebLLMModel['stream']>
    next: () => Promise<BrowserPacket<ModelStreamEvent, WebLLMChunk>>
    pending?: Promise<BrowserPacket<ModelStreamEvent, WebLLMChunk>>
    controller: AbortController
    closing?: Promise<void>
    takeChunks?: () => WebLLMChunk[]
  }
  const calls = new Map<string, Call>()
  const nativeCalls = new Set<Promise<unknown>>()
  let closed = false
  function checkOpen() {
    if (closed) throw new Error('The browser bridge is closed.')
  }
  const bridge: BrowserBridge = {
    async withEngine(run) {
      checkOpen()
      const pending = withEngine(engine, (engine) => {
        checkOpen()
        return run(engine)
      })
      nativeCalls.add(pending)
      try {
        return await pending
      } finally {
        nativeCalls.delete(pending)
      }
    },
    async start({ messages, config, options, includeChunks }: BrowserRequest) {
      checkOpen()
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
        const pending = call.pending ?? call.next()
        // Reserve the next event before waiting, so overlapping reads cannot share it.
        call.pending = pending.then((result) => (result.done ? result : call.next()))
        void call.pending.catch(() => {})
        const result = await pending
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
    stop(id: string): Promise<void> {
      const call = calls.get(id)
      if (!call) return Promise.resolve()
      if (call.closing) return call.closing
      call.controller.abort()
      call.closing = (async () => {
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
        ...[...calls.keys()].map((id) => bridge.stop(id)),
        // Native errors go to their caller; close waits for their cleanup without replaying them.
        ...Array.from(nativeCalls, (call) => call.catch(() => {})),
      ])
      for (const result of results) if (result.status === 'rejected') throw result.reason
    },
  }
  return bridge
}
