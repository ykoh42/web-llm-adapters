import { ModelError } from '@strands-agents/sdk'
import type { ChatCompletionRequestStreaming, MLCEngineInterface } from '@mlc-ai/web-llm'

interface ActiveCall {
  signal: AbortSignal | undefined
  closed: Promise<void>
}

const activeCalls = new WeakMap<MLCEngineInterface, ActiveCall>()

/** Own the native stream until it closes, even if Strands abandons its iterator on abort. */
export async function startCall(
  engine: MLCEngineInterface,
  request: ChatCompletionRequestStreaming,
  signal?: AbortSignal,
) {
  for (;;) {
    const previous = activeCalls.get(engine)
    if (!previous) break
    if (!previous.signal?.aborted) throw new ModelError('This WebLLM engine is already in use.')
    await previous.closed
  }
  signal?.throwIfAborted()
  let release!: () => void
  const closed = new Promise<void>((resolve) => {
    release = resolve
  })
  activeCalls.set(engine, { signal, closed })

  const source = Promise.resolve()
    .then(() => engine.chat.completions.create(request))
    .then((chunks) => chunks[Symbol.asyncIterator]())
  let closing: Promise<void> | undefined
  function close(interrupt: boolean) {
    closing ??= (async () => {
      try {
        if (interrupt) engine.interruptGenerate()
        const iterator = await source
        await iterator.return?.()
      } finally {
        signal?.removeEventListener('abort', cancel)
        activeCalls.delete(engine)
        release()
      }
    })()
    return closing
  }
  function cancel() {
    // close() keeps the rejection for a consuming caller. An abandoned iterator has no caller.
    void close(true).catch(() => {})
  }
  signal?.addEventListener('abort', cancel, { once: true })
  if (signal?.aborted) cancel()
  try {
    const iterator = await source
    signal?.throwIfAborted()
    return { chunks: { [Symbol.asyncIterator]: () => iterator }, close }
  } catch (error) {
    await close(false).catch(() => {})
    throw error
  }
}
