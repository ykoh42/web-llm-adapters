/** Cancellation takes precedence over an error after the last emitted chunk. */
export function throwStreamError(
  failure: { error: unknown } | undefined,
  signal?: AbortSignal,
): void {
  if (failure && !signal?.aborted) throw failure.error
}
