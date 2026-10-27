const adapters = ['langchain', 'pi', 'strands'] as const
const features = ['tools', 'thinking', 'image', 'audio', 'video'] as const
export type Feature =
  | (typeof features)[number]
  | 'constrained-tools'
  | 'tool-protocol'
  | 'embeddings'
  | 'cancel'
  | 'concurrency'

function names(value: string) {
  if (value === 'All') return [...adapters]
  const names = value.toLowerCase().split(/,\s*/)
  if (!names.length || names.some((name) => !adapters.some((adapter) => adapter === name)))
    throw new Error(`Unknown adapters: ${value}`)
  return names
}

export function supportClaims(markdown: string) {
  const claims: string[] = []
  const section = markdown.split(/^## Models\r?$/m)[1]?.split(/^## /m)[0]
  if (!section) throw new Error('Missing Models section.')
  for (const line of section.split('\n')) {
    const cells = line
      .split('|')
      .slice(1, -1)
      .map((cell) => cell.trim())
    const model = /^`([^`]+)`$/.exec(cells[0] ?? '')?.[1]
    if (!model) continue
    if (cells.length === 9) {
      const verified = names(cells[8]!)
      for (const [index, feature] of features.entries()) {
        const cell = cells[index + 3]!
        if (cell === 'X') continue
        const match = /^O(\*)?(?: \(([^)]+)\))?$/.exec(cell)
        if (!match) throw new Error(`Unknown support value: ${cell}`)
        if (match[1] && feature !== 'tools')
          throw new Error('Only tool checks can use constrained generation.')
        for (const adapter of match[2] ? names(match[2]) : verified) {
          if (!verified.includes(adapter)) throw new Error(`Unexpected adapter: ${adapter}`)
          const checks = match[1] ? ['constrained-tools', 'tool-protocol'] : [feature]
          for (const check of checks) claims.push(`${adapter}/${model}/${check}`)
        }
      }
    } else if (cells.length === 3) {
      const feature =
        cells[1] === 'Image input'
          ? 'image'
          : /^Embeddings, \d+ dimensions$/.test(cells[1]!)
            ? 'embeddings'
            : undefined
      if (!feature) throw new Error(`Unknown model use: ${cells[1]}`)
      for (const adapter of names(cells[2]!)) claims.push(`${adapter}/${model}/${feature}`)
    } else {
      throw new Error(`Unknown model row: ${line}`)
    }
  }
  if (!claims.length || new Set(claims).size !== claims.length)
    throw new Error('Support table must contain unique model checks.')
  return [
    ...claims,
    ...adapters.flatMap((adapter) =>
      ['cancel', 'concurrency'].map((feature) => `${adapter}/engine/${feature}`),
    ),
  ].sort()
}

export function modelCheck(model: string, ...features: Feature[]) {
  return {
    tag: '@release',
    annotation: features.map((feature) => ({
      type: 'supports',
      description: `${model}/${feature}`,
    })),
  }
}
