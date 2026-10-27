import type { Api, Model } from '@earendil-works/pi-ai'
import type { ProtocolInput } from '../../../shared/web-llm/protocol.js'
import type { WebLLMStreamOptions } from './types.js'
import { waitForCallback } from '../../../shared/web-llm/callback.js'

export function streamHooks(
  model: Model<Api>,
  modelId: string,
  call: WebLLMStreamOptions,
  signal?: AbortSignal,
): Pick<ProtocolInput, 'prepare' | 'open'> {
  const hooks: Pick<ProtocolInput, 'prepare' | 'open'> = {}
  if (call.onPayload)
    hooks.prepare = async (payload) => {
      const returned = await waitForCallback(
        Promise.resolve(call.onPayload?.(payload, model)),
        signal,
      )
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
    }
  if (call.onProviderStreamEvent)
    hooks.open = async (run) => {
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
                await waitForCallback(
                  Promise.resolve(call.onProviderStreamEvent?.(structuredClone(item.value), model)),
                  signal,
                )
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
    }
  return hooks
}
