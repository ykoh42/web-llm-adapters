/** Keep serialized error fields intact across browser transports. */
export function readBrowserError<T>(result: { error: T; readonly errorJSON?: string }): T {
  return result.errorJSON === undefined ? result.error : JSON.parse(result.errorJSON)
}
