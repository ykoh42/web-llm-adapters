import { waitForCallback } from '../web-llm/callback.js'

/** A cancelled browser start must still clean up if its ID arrives later. */
export async function startBrowserCall(
  start: Promise<string>,
  stop: (id: string) => Promise<void>,
  signal?: AbortSignal,
): Promise<string> {
  try {
    return await waitForCallback(start, signal)
  } catch (error) {
    if (signal?.aborted) void start.then(stop).catch(() => {})
    throw error
  }
}
