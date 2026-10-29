type Piece = { value: unknown } | { text: string }

/** Preserve negative zero in values parsed from the model's tool call. */
export function toolJSON(value: unknown, source: string): string {
  if (!source.includes('-') || !hasNegativeZero(value)) return JSON.stringify(value)

  const output: string[] = []
  const pending: Piece[] = [{ value }]
  while (pending.length) {
    const item = pending.pop()!
    if ('text' in item) {
      output.push(item.text)
      continue
    }
    const value = item.value
    if (Object.is(value, -0)) output.push('-0')
    else if (Array.isArray(value)) writeArray(value, output, pending)
    else if (value && typeof value === 'object') writeObject(value, output, pending)
    else output.push(JSON.stringify(value))
  }
  return output.join('')
}

function writeArray(value: unknown[], output: string[], pending: Piece[]): void {
  output.push('[')
  pending.push({ text: ']' })
  for (let index = value.length - 1; index >= 0; index--) {
    pending.push({ value: value[index] })
    if (index) pending.push({ text: ',' })
  }
}

function writeObject(value: object, output: string[], pending: Piece[]): void {
  const entries = Object.entries(value)
  output.push('{')
  pending.push({ text: '}' })
  for (let index = entries.length - 1; index >= 0; index--) {
    const [key, value] = entries[index]!
    pending.push({ value }, { text: `${JSON.stringify(key)}:` })
    if (index) pending.push({ text: ',' })
  }
}

export function hasNegativeZero(value: unknown): boolean {
  if (Object.is(value, -0)) return true
  if (!value || typeof value !== 'object') return false
  const pending: object[] = [value]
  while (pending.length) {
    const item = pending.pop()!
    for (const value of Array.isArray(item) ? item : Object.values(item)) {
      if (Object.is(value, -0)) return true
      if (value && typeof value === 'object') pending.push(value)
    }
  }
  return false
}
