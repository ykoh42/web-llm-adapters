/** JSON preserves reserved keys, while these paths preserve negative zero. */
export function negativeZeros(value: unknown): string[][] {
  const paths: string[][] = []
  const pending: { value: unknown; path: string[] }[] = [{ value, path: [] }]
  while (pending.length) {
    const { value, path } = pending.pop()!
    if (Object.is(value, -0)) {
      paths.push(path)
      continue
    }
    if (!value || typeof value !== 'object') continue
    const entries = Array.isArray(value) ? value.entries() : Object.entries(value)
    for (const [key, item] of entries) {
      if (Object.is(item, -0)) paths.push([...path, String(key)])
      else if (item && typeof item === 'object')
        pending.push({ value: item, path: [...path, String(key)] })
    }
  }
  return paths
}

export function restoreNegativeZeros(value: unknown, paths: string[][]): void {
  for (const path of paths) {
    let parent = value as Record<string, unknown>
    for (const key of path.slice(0, -1)) parent = parent[key] as Record<string, unknown>
    Object.defineProperty(parent, path.at(-1)!, {
      value: -0,
      configurable: true,
      enumerable: true,
      writable: true,
    })
  }
}
