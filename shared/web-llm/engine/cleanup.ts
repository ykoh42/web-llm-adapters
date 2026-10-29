const cleanups = new WeakMap<AbortSignal, Promise<void>>()

/** A browser bridge can wait for the cleanup of its own cancelled call. */
export function trackCleanup(signal: AbortSignal, cleanup: Promise<void>): void {
  cleanups.set(signal, cleanup)
}

export async function waitForCleanup(signal: AbortSignal): Promise<void> {
  try {
    await cleanups.get(signal)
  } catch (error) {
    if (error !== signal.reason) throw error
  }
}
