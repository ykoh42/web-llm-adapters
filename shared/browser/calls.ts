import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { withEngine } from '../web-llm/engine/access.js'

export interface BrowserCall<Read extends { done?: boolean }> {
  stream: AsyncGenerator<unknown>
  next: () => Promise<Read>
  pending?: Promise<Read>
  controller: AbortController
  closing?: Promise<void>
}

/** Manage browser calls without taking ownership of the caller's engine. */
export function browserCalls<Read extends { done?: boolean }, Call extends BrowserCall<Read>>(
  engine: MLCEngineInterface,
) {
  const calls = new Map<string, Call>()
  const nativeCalls = new Set<Promise<unknown>>()
  let closed = false

  function checkOpen() {
    if (closed) throw new Error('The browser bridge is closed.')
  }

  async function track<T>(call: Promise<T>) {
    nativeCalls.add(call)
    try {
      return await call
    } finally {
      nativeCalls.delete(call)
    }
  }

  async function run<T>(callback: (engine: MLCEngineInterface) => T | Promise<T>) {
    checkOpen()
    return track(
      withEngine(engine, (engine) => {
        checkOpen()
        return callback(engine)
      }),
    )
  }

  function read(call: Call): Promise<Read> {
    const pending = call.pending ?? call.next()
    // Reserve the next result before waiting so overlapping reads stay separate.
    call.pending = pending.then((result) => (result.done ? result : call.next()))
    void call.pending.catch(() => {})
    return pending
  }

  function stop(id: string): Promise<void> {
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
  }

  async function close(stopCall: (id: string) => Promise<void>) {
    closed = true
    const results = await Promise.allSettled([
      ...Array.from(calls.keys(), (id) => stopCall(id)),
      // Native errors belong to their callers; close only waits for cleanup.
      ...Array.from(nativeCalls, (call) => call.catch(() => {})),
    ])
    for (const result of results) if (result.status === 'rejected') throw result.reason
  }

  return { calls, checkOpen, track, withEngine: run, read, stop, close }
}
