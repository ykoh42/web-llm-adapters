export const messageBoundaryTokens = ['<|im_start|>', '<|im_end|>', '<|endoftext|>'] as const
const markers = ['<parameter=', '</parameter>', ...messageBoundaryTokens]
let grammar: string | undefined

export function containsMessageBoundaryToken(text: string) {
  return messageBoundaryTokens.some((token) => text.includes(token))
}

/** Match raw parameter text without admitting parser delimiters or a trailing CR. */
export function xmlStringFormat() {
  grammar ??= stringGrammar()
  return { type: 'grammar', grammar }
}

function stringGrammar() {
  // Each rule tracks a marker prefix. Completing a forbidden marker has no transition.
  const prefixes = [
    ...new Set([
      '',
      '\r',
      ...markers.flatMap((marker) =>
        Array.from({ length: marker.length - 1 }, (_, index) => marker.slice(0, index + 1)),
      ),
    ]),
  ]
  const alphabet = [...new Set(markers.join('') + '\r')]
  const name = (index: number) => (index === 0 ? 'root' : `s${index}`)
  return prefixes
    .map((prefix, index) => {
      const transitions = new Map<number, string[]>()
      const excluded = []
      for (const character of alphabet) {
        const text = prefix + character
        if (markers.some((marker) => text.endsWith(marker))) {
          excluded.push(character)
          continue
        }
        const next = prefixes.reduce(
          (longest, candidate, position) =>
            text.endsWith(candidate) && candidate.length > prefixes[longest]!.length
              ? position
              : longest,
          0,
        )
        if (next === 0) continue
        excluded.push(character)
        const characters = transitions.get(next) ?? []
        characters.push(character)
        transitions.set(next, characters)
      }
      const alternatives = [
        ...(!prefix.endsWith('\r') ? ['""'] : []),
        `[^${characters(excluded)}] root`,
        ...[...transitions].map(([next, values]) => `[${characters(values)}] ${name(next)}`),
      ]
      return `${name(index)} ::= ${alternatives.join(' | ')}`
    })
    .join('\n')
}

function characters(values: string[]) {
  return values
    .map((value) => (value === '\r' ? '\\r' : value.replace(/[\\\]\-^]/g, '\\$&')))
    .join('')
}
