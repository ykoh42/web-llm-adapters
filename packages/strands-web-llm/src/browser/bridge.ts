import { Message } from '@strands-agents/sdk'
import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { WebLLMModel } from '../model.js'
import { withEngine } from '../engine.js'
import type { WebLLMChunk } from '../types.js'
import type { BrowserBridge, BrowserRequest, BrowserResult, BrowserCountRequest } from './types.js'

/** Serve a loaded engine on a private browser page. The caller owns its lifetime. */
export function createBrowserBridge(engine: MLCEngineInterface): BrowserBridge {
  interface Call {
    stream: ReturnType<WebLLMModel['stream']>
    controller: AbortController
    closing?: Promise<void>
    chunks: WebLLMChunk[]
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
    async count({ messages, config, options }: BrowserCountRequest) {
      checkOpen()
      const model = new WebLLMModel({ engine, ...config })
      return model.countTokens(messages.map(Message.fromJSON), options)
    },
    async start({ messages, config, options, includeChunks }: BrowserRequest) {
      checkOpen()
      const id = crypto.randomUUID()
      const controller = new AbortController()
      const chunks: WebLLMChunk[] = []
      const model = new WebLLMModel({
        engine,
        ...config,
        ...(includeChunks && { onChunk: (chunk: WebLLMChunk) => chunks.push(chunk) }),
      })
      const stream = model.stream(messages.map(Message.fromJSON), {
        ...options,
        cancelSignal: controller.signal,
      })
      calls.set(id, { stream, controller, chunks })
      return id
    },
    async next(id: string): Promise<BrowserResult> {
      const call = calls.get(id)
      try {
        if (!call) throw new Error('Unknown browser call.')
        const result = await call.stream.next()
        if (result.done) calls.delete(id)
        return { ...result, ...(call.chunks.length && { chunks: call.chunks.splice(0) }) }
      } catch (error) {
        calls.delete(id)
        return {
          ...(call?.chunks.length && { chunks: call.chunks.splice(0) }),
          error: {
            name: error instanceof Error ? error.name : 'Error',
            message: error instanceof Error ? error.message : String(error),
          },
        }
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
