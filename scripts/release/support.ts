import type { ChatCompletionRequestBase, CompletionCreateParamsBase } from '@mlc-ai/web-llm'

const adapters = ['langchain', 'pi', 'strands'] as const
const features = ['tools', 'thinking', 'image', 'audio', 'video'] as const
export type Feature =
  | (typeof features)[number]
  | 'text'
  | 'constrained-tools'
  | 'tool-protocol'
  | 'embeddings'
  | 'cancel'
  | 'concurrency'
  | 'resume'

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
  for (const { line, cells } of modelRows(section)) {
    const model = /^`([^`]+)`$/.exec(cells[0] ?? '')?.[1]
    if (!model) throw new Error('A model row needs a backtick-quoted ID.')
    if (cells.length === 9) {
      addChatClaims(claims, model, cells)
      continue
    }
    if (cells.length === 3) {
      addModelClaims(claims, model, cells)
      continue
    }
    throw new Error(`Unknown model row: ${line}`)
  }
  if (!claims.length || new Set(claims).size !== claims.length)
    throw new Error('Support table must contain unique model checks.')
  return [
    ...claims,
    ...adapters.flatMap((adapter) =>
      ['cancel', 'concurrency', 'resume'].map((feature) => `${adapter}/engine/${feature}`),
    ),
  ].sort()
}

function* modelRows(section: string) {
  const lines = section.split('\n')
  let inTable = false
  for (const [index, line] of lines.entries()) {
    const cells = tableCells(line)
    if (!cells.length) {
      inTable = false
      continue
    }
    if (isDivider(cells) || isDivider(tableCells(lines[index + 1] ?? ''))) {
      inTable = true
      continue
    }
    if (!inTable && !line.trimStart().startsWith('|')) continue
    inTable = true
    yield { line, cells }
  }
}

function addChatClaims(claims: string[], model: string, cells: string[]) {
  const verified = names(cells[8]!)
  const covered = new Set<string>()
  for (const [index, feature] of features.entries()) {
    const support = readSupport(cells[index + 3]!, feature, verified)
    if (!support) continue
    for (const adapter of support.adapters) {
      if (!verified.includes(adapter)) throw new Error(`Unexpected adapter: ${adapter}`)
      covered.add(adapter)
      for (const check of support.checks) claims.push(`${adapter}/${model}/${check}`)
    }
  }
  for (const adapter of verified) if (!covered.has(adapter)) claims.push(`${adapter}/${model}/text`)
}

function readSupport(cell: string, feature: (typeof features)[number], verified: string[]) {
  if (cell === 'X') return
  const match = /^O(\*)?(?: \(([^)]+)\))?$/.exec(cell)
  if (!match) throw new Error(`Unknown support value: ${cell}`)
  if (match[1] && feature !== 'tools')
    throw new Error('Only tool checks can use constrained generation.')
  return {
    adapters: match[2] ? names(match[2]) : verified,
    checks: match[1] ? ['constrained-tools', 'tool-protocol'] : [feature],
  }
}

function addModelClaims(claims: string[], model: string, cells: string[]) {
  const feature =
    cells[1] === 'Image input'
      ? 'image'
      : /^Embeddings, \d+ dimensions$/.test(cells[1]!)
        ? 'embeddings'
        : undefined
  if (!feature) throw new Error(`Unknown model use: ${cells[1]}`)
  for (const adapter of names(cells[2]!)) claims.push(`${adapter}/${model}/${feature}`)
}

function tableCells(line: string) {
  const row = line.trim()
  if (!row.includes('|')) return []
  return row
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((cell) => cell.trim())
}

function isDivider(cells: string[]) {
  return cells.length > 1 && cells.every((cell) => /^:?-{3,}:?$/.test(cell))
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

/** Count reasoning only when it was observed and passed into the next model call. */
export function recordThinking(
  info: { annotations: { type: string; description?: string }[] },
  model: string,
  thoughts: readonly string[],
  requests: readonly (ChatCompletionRequestBase | CompletionCreateParamsBase)[],
) {
  let observed = false
  for (const [index, thought] of thoughts.entries()) {
    const previous = requests[index]
    const next = requests[index + 1]
    if (!previous || !next || !/\S/.test(thought)) continue
    if (model.startsWith('gemma-4-')) {
      checkGemmaThinking(previous, next, thought)
    } else if (model.startsWith('Qwen3')) {
      checkQwenThinking(previous, next, thought, model)
    } else continue
    observed = true
  }
  if (observed) info.annotations.push({ type: 'supports', description: `${model}/thinking` })
}

type Request = ChatCompletionRequestBase | CompletionCreateParamsBase

function checkGemmaThinking(previous: Request, next: Request, thought: string) {
  const block = `<|channel>thought\n${thought}\n<channel|>`
  if (
    !('prompt' in previous) ||
    !('prompt' in next) ||
    occurrences(next.prompt, block) <= occurrences(previous.prompt, block)
  )
    throw new Error('The next request lost the native thinking block.')
}

function checkQwenThinking(previous: Request, next: Request, thought: string, model: string) {
  const content = model.startsWith('Qwen3.5-') ? thought.trim() : thought.replace(/^\n+|\n+$/g, '')
  const block = `<think>\n${content}\n</think>`
  if (
    !('messages' in previous) ||
    !('messages' in next) ||
    assistantOccurrences(next.messages, block) <= assistantOccurrences(previous.messages, block)
  )
    throw new Error('The next request lost the native thinking block.')
}

function occurrences(text: string, block: string) {
  let count = 0
  let index = 0
  while ((index = text.indexOf(block, index)) !== -1) {
    count++
    index += block.length
  }
  return count
}

function assistantOccurrences(messages: ChatCompletionRequestBase['messages'], block: string) {
  return messages.reduce(
    (count, message) =>
      count +
      (message.role === 'assistant' && typeof message.content === 'string'
        ? occurrences(message.content, block)
        : 0),
    0,
  )
}
