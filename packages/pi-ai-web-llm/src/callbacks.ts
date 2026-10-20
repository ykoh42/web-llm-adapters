import type { Api, Model } from '@earendil-works/pi-ai'
import type { ProtocolInput } from '../../../shared/web-llm/protocol.js'
import type { WebLLMStreamOptions } from './types.js'

export function streamHooks(
  model: Model<Api>,
  modelId: string,
  call: WebLLMStreamOptions,
  signal?: AbortSignal,
): Pick<ProtocolInput, 'prepare' | 'open'> {
  return {
    prepare: async (payload) => {
      const returned = await call.onPayload?.(payload, model)
      const changed = returned === undefined ? payload : returned
      signal?.throwIfAborted()
      if (
        typeof changed !== 'object' ||
        changed === null ||
        !('model' in changed) ||
        changed.model !== modelId ||
        !('stream' in changed) ||
        changed.stream !== true ||
        !('n' in changed) ||
        changed.n !== 1
      )
        throw new Error('onPayload must preserve the loaded model and one streaming response.')
      return changed as typeof payload
    },
    open: async (run) => {
      const chunks = await run()
      if (!call.onProviderStreamEvent) return chunks
      return {
        [Symbol.asyncIterator]() {
          const iterator = chunks[Symbol.asyncIterator]()
          return {
            async next() {
              const item = await iterator.next()
              if (!item.done) {
                signal?.throwIfAborted()
                await call.onProviderStreamEvent?.(item.value, model)
                signal?.throwIfAborted()
              }
              return item
            },
            // Forward cleanup even if cancellation arrives before the first next().
            async return() {
              return iterator.return ? iterator.return() : { done: true as const, value: undefined }
            },
          }
        },
      }
    },
  }
}
