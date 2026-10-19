import { ModelError, type ToolSpec } from '../types.js'
import type { ChatCompletionChunk } from '@mlc-ai/web-llm'
import { stopReason, type StreamChunk, type ToolDelta } from '../stream/chunks.js'
import { textEnd } from '../gemma/markers.js'

const thinkStart = '<think>'
const thinkEnd = '</think>'
const callStart = '<tool_call>'
const callEnd = '</tool_call>'

/** Split Qwen3.5 reasoning and parse its XML function and parameter tags. */
export async function* qwen35Chunks(
  chunks: AsyncIterable<ChatCompletionChunk>,
  tools: ToolSpec[],
  onChunk?: (chunk: ChatCompletionChunk) => void,
  parseProtocol = true,
): AsyncGenerator<StreamChunk> {
  let mode: 'text' | 'thinking' | 'tool' = 'text'
  let pending = ''
  let toolText = ''
  let callIndex = 0
  for await (const chunk of chunks) {
    onChunk?.(chunk)
    const choice = chunk.choices[0]
    if (chunk.choices.length > 1 || (choice && choice.index !== 0))
      throw new ModelError('WebLLM adapters need one response per request.')
    pending += choice?.delta.content ?? ''
    let text = ''
    let reasoning = ''
    const calls: ToolDelta[] = []

    while (pending) {
      if (mode === 'tool') {
        const at = pending.indexOf(callEnd)
        if (at === -1) {
          const end = textEnd(pending, [callEnd])
          toolText += pending.slice(0, end)
          pending = pending.slice(end)
          break
        }
        toolText += pending.slice(0, at)
        pending = pending.slice(at + callEnd.length)
        const parsed = parseCalls(toolText, tools, callIndex)
        calls.push(...parsed)
        callIndex += parsed.length
        toolText = ''
        mode = 'text'
        continue
      }

      if (mode === 'text' && !parseProtocol) {
        text += pending
        pending = ''
        break
      }

      const markers = mode === 'thinking' ? [thinkEnd] : [thinkStart, callStart]
      const at = earliest(pending, markers)
      if (at !== undefined) {
        const value = pending.slice(0, at.index)
        if (mode === 'thinking') reasoning += value
        else text += value
        pending = pending.slice(at.index + at.marker.length)
        mode = mode === 'thinking' ? 'text' : at.marker === thinkStart ? 'thinking' : 'tool'
        continue
      }
      const end = textEnd(pending, markers)
      const value = pending.slice(0, end)
      if (mode === 'thinking') reasoning += value
      else text += value
      pending = pending.slice(end)
      break
    }

    const stop = stopReason(choice?.finish_reason)
    if (stop) {
      if (mode === 'tool' && stop !== 'maxTokens' && stop !== 'cancelled')
        throw new ModelError('Qwen3.5 ended inside a tool call.')
      if (mode === 'tool') {
        pending = ''
        toolText = ''
      }
      if (mode === 'thinking') {
        reasoning += pending
        pending = ''
      } else {
        text += pending
        pending = ''
      }
      mode = 'text'
    }
    yield { text, reasoning, toolText: '', calls, stop, usage: chunk.usage }
  }
}

function earliest(text: string, markers: string[]) {
  let found: { index: number; marker: string } | undefined
  for (const marker of markers) {
    const index = text.indexOf(marker)
    if (index !== -1 && (!found || index < found.index)) found = { index, marker }
  }
  return found
}

function parseCalls(source: string, tools: ToolSpec[], index: number): ToolDelta[] {
  const calls: ToolDelta[] = []
  const functions = /<function=([^>\s]+)>([\s\S]*?)<\/function>/g
  for (const match of source.matchAll(functions)) {
    const name = match[1]!
    const tool = tools.find((item) => item.name === name)
    if (!tool) throw new ModelError(`Qwen3.5 returned unknown tool "${name}".`)
    const input = Object.create(null) as Record<string, unknown>
    const body = match[2]!
    const parameters = /<parameter=([^>\s]+)>([\s\S]*?)(?:<\/parameter>|(?=<parameter=))/g
    const values = [...body.matchAll(parameters)]
    // Like vLLM's Qwen parser, ignore redundant closers outside complete parameters.
    if (body.replace(parameters, '').replaceAll('</parameter>', '').trim())
      throw new ModelError('Qwen3.5 returned a malformed tool parameter.')
    for (const parameter of values) {
      const key = parameter[1]!
      if (Object.hasOwn(input, key)) throw new ModelError(`Qwen3.5 repeated parameter "${key}".`)
      input[key] = parseValue(parameter[2]!, propertyTypes(tool, key))
    }
    calls.push({
      index: index + calls.length,
      id: crypto.randomUUID(),
      name,
      input: JSON.stringify(input),
    })
  }
  if (!calls.length || source.replace(functions, '').trim())
    throw new ModelError('Qwen3.5 returned a malformed tool call.')
  return calls
}

function parseValue(source: string, types: string[] | undefined) {
  const text = source.replace(/^\r?\n/, '').replace(/\r?\n$/, '')
  if (types?.includes('string')) {
    if (types.includes('null') && text.trim() === 'null') return null
    const otherTypes = types.filter((type) => type !== 'string' && type !== 'null')
    if (otherTypes.length) {
      try {
        const value: unknown = JSON.parse(text)
        if (otherTypes.some((type) => matchesType(value, type)))
          throw new ModelError('Qwen3.5 returned an ambiguous tool argument.')
      } catch (cause) {
        if (cause instanceof ModelError) throw cause
      }
    }
    return text
  }
  try {
    const value: unknown = JSON.parse(text)
    if (types?.length && !types.some((type) => matchesType(value, type)))
      throw new ModelError('Qwen3.5 returned a tool argument with the wrong type.')
    return value
  } catch (cause) {
    if (types === undefined) return text
    if (cause instanceof ModelError) throw cause
    throw new ModelError('Qwen3.5 returned an invalid tool argument.', { cause })
  }
}

function propertyTypes(tool: ToolSpec, name: string): string[] | undefined {
  const schema = tool.inputSchema
  if (!isRecord(schema)) return undefined
  const schemas = propertySchemas(schema, name, schema)
  const types = schemas
    .map((value) => readTypes(value, schema))
    .filter((value): value is string[] => value !== undefined)
  if (!types.length) return undefined

  const [first, ...rest] = types
  const intersection = rest.reduce(intersectTypes, first!)
  if (!intersection.length) throw new ModelError('Qwen3.5 found incompatible tool argument types.')
  return intersection
}

function intersectTypes(types: string[], next: string[]) {
  const result = new Set<string>()
  for (const type of types) {
    for (const candidate of next) {
      if (type === candidate) result.add(type)
      else if (type === 'integer' && candidate === 'number') result.add('integer')
      else if (type === 'number' && candidate === 'integer') result.add('integer')
    }
  }
  return [...result]
}

function propertySchemas(
  schema: Record<string, unknown>,
  name: string,
  root: Record<string, unknown>,
  references = new Set<string>(),
): unknown[] {
  const schemas: unknown[] = []

  if (typeof schema.$ref === 'string') {
    if (references.has(schema.$ref)) throw new ModelError('Qwen3.5 found a recursive tool schema.')
    const target = resolveReference(root, schema.$ref)
    if (!isRecord(target)) throw new ModelError('Qwen3.5 cannot resolve a tool schema reference.')
    schemas.push(...propertySchemas(target, name, root, new Set([...references, schema.$ref])))
  }

  if (isRecord(schema.properties) && Object.hasOwn(schema.properties, name)) {
    schemas.push(schema.properties[name])
  } else if (schema.additionalProperties === false) {
    throw new ModelError(`Qwen3.5 returned unknown tool parameter "${name}".`)
  } else if (isRecord(schema.additionalProperties)) {
    schemas.push(schema.additionalProperties)
  }

  if (Array.isArray(schema.allOf)) {
    for (const branch of schema.allOf) {
      if (!isRecord(branch)) throw new ModelError('Qwen3.5 cannot read a tool schema.')
      schemas.push(...propertySchemas(branch, name, root, references))
    }
  }

  if (schema.anyOf !== undefined || schema.oneOf !== undefined)
    throw new ModelError('Qwen3.5 cannot determine a tool argument type.')

  return schemas
}

function readTypes(
  value: unknown,
  root: Record<string, unknown>,
  references = new Set<string>(),
): string[] | undefined {
  if (!isRecord(value)) return undefined

  const constraints: string[][] = []
  if (typeof value.$ref === 'string') {
    if (references.has(value.$ref)) throw new ModelError('Qwen3.5 found a recursive tool schema.')
    const target = resolveReference(root, value.$ref)
    if (!isRecord(target)) throw new ModelError('Qwen3.5 cannot resolve a tool schema reference.')
    const types = readTypes(target, root, new Set([...references, value.$ref]))
    if (types) constraints.push(types)
  }

  const types = Array.isArray(value.type)
    ? value.type.filter((type): type is string => typeof type === 'string')
    : typeof value.type === 'string'
      ? [value.type]
      : []
  if (types.length) constraints.push(types)

  for (const keyword of ['anyOf', 'oneOf'] as const) {
    const alternatives = value[keyword]
    if (alternatives === undefined) continue
    if (!Array.isArray(alternatives) || !alternatives.length)
      throw new ModelError('Qwen3.5 cannot determine a tool argument type.')
    const branchTypes = alternatives.map((branch) => readTypes(branch, root, references))
    if (branchTypes.some((branch) => !branch))
      throw new ModelError('Qwen3.5 cannot determine a tool argument type.')
    const options = branchTypes as string[][]
    if (keyword === 'oneOf' && hasOverlappingTypes(options))
      throw new ModelError('Qwen3.5 cannot determine a tool argument type.')
    constraints.push([...new Set(options.flat())])
  }

  if (value.allOf !== undefined) {
    if (!Array.isArray(value.allOf))
      throw new ModelError('Qwen3.5 cannot determine a tool argument type.')
    const branchTypes = value.allOf
      .map((branch) => readTypes(branch, root, references))
      .filter((branch): branch is string[] => branch !== undefined)
    constraints.push(...branchTypes)
  }

  if (!constraints.length) return undefined
  const [first, ...rest] = constraints
  const intersection = rest.reduce(intersectTypes, first!)
  if (!intersection.length) throw new ModelError('Qwen3.5 found incompatible tool argument types.')
  return intersection
}

function hasOverlappingTypes(options: string[][]) {
  return options.some((types, index) =>
    options.slice(index + 1).some((next) => intersectTypes(types, next).length > 0),
  )
}

function resolveReference(root: Record<string, unknown>, reference: string): unknown {
  if (!reference.startsWith('#/')) return undefined
  return reference
    .slice(2)
    .split('/')
    .map((part) => decodeURIComponent(part.replaceAll('~1', '/').replaceAll('~0', '~')))
    .reduce<unknown>((value, part) => (isRecord(value) ? value[part] : undefined), root)
}

function matchesType(value: unknown, type: string) {
  if (type === 'integer') return typeof value === 'number' && Number.isInteger(value)
  if (type === 'null') return value === null
  if (type === 'array') return Array.isArray(value)
  if (type === 'object') return isRecord(value)
  return typeof value === type
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
