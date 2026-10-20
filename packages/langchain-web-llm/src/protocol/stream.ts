import type { AsyncCaller } from '@langchain/core/utils/async_caller'
import {
  protocolStream as nativeStream,
  type ProtocolInput,
} from '../../../../shared/web-llm/protocol.js'
import { modelError } from '../error.js'

export function protocolStream({
  caller,
  maxRetries,
  ...input
}: Omit<ProtocolInput, 'open'> & {
  caller: AsyncCaller
  maxRetries: number | undefined
}) {
  const { signal } = input
  const open = <T>(run: () => Promise<AsyncIterable<T>>) => {
    let attempt: Promise<AsyncIterable<T>> | undefined
    const pending = caller.callWithOptions(
      {
        ...(maxRetries !== undefined && { maxRetries }),
        ...(signal && { signal }),
      },
      async () => {
        signal?.throwIfAborted()
        try {
          attempt = run()
          return await attempt
        } catch (error) {
          throw modelError(error)
        }
      },
    )
    if (!signal) return pending
    return pending.catch(async (error) => {
      if (signal.aborted && attempt) await closeUnclaimedStream(attempt).catch(() => {})
      throw error
    })
  }
  return nativeStream({ ...input, open })
}

async function closeUnclaimedStream<T>(pending: Promise<AsyncIterable<T>>) {
  let stream: AsyncIterable<T>
  try {
    stream = await pending
  } catch {
    return
  }

  await stream[Symbol.asyncIterator]().return?.()
}
