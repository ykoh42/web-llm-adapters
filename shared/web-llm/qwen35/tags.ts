import type { ToolSpec } from '../types.js'
import { isJsonArgument, type ArgumentSchemaCache } from './schema.js'

const markers = ['<function=', '</function>', '<parameter=', '</parameter>', '</tool_call>']

/** Scan XML boundaries while keeping typed JSON containers intact across chunks. */
export class QwenTags {
  static readonly markers = markers
  inFunction = false
  inParameter = false
  private header: { type: 'function' | 'parameter'; name: string } | undefined
  private parameter = ''
  private valueStart = false
  private depth = 0
  private quoted = false
  private escaped = false
  private tool: ToolSpec | undefined
  private readonly tools: ToolSpec[]
  private readonly schemas: ArgumentSchemaCache

  constructor(tools: ToolSpec[], schemas: ArgumentSchemaCache = new WeakMap()) {
    this.tools = tools
    this.schemas = schemas
  }

  *read(text: string, limit = text.length) {
    for (let index = 0; index < limit; index++) {
      const character = text[index]!
      if (this.header) {
        const header = this.header
        if (character !== '>') header.name += character
        else {
          if (header.type === 'function')
            this.tool = this.tools.find((tool) => tool.name === header.name)
          else {
            this.parameter = header.name
            this.valueStart = true
          }
          this.header = undefined
        }
        continue
      }
      if (this.valueStart) {
        if (/\s/.test(character)) continue
        this.valueStart = false
        if (
          (character === '{' || character === '[') &&
          this.tool &&
          isJsonArgument(this.tool, this.parameter, this.schemas)
        ) {
          this.depth = 1
          continue
        }
      }
      if (this.depth) {
        if (this.quoted) {
          if (this.escaped) this.escaped = false
          else if (character === '\\') this.escaped = true
          else if (character === '"') this.quoted = false
        } else if (character === '"') this.quoted = true
        else if (character === '{' || character === '[') this.depth++
        else if (character === '}' || character === ']') this.depth--
        continue
      }
      if (character !== '<') {
        const next = text.indexOf('<', index + 1)
        if (next === -1 || next >= limit) break
        index = next - 1
        continue
      }
      const marker = markers.find((marker) => text.startsWith(marker, index))
      if (!marker) continue
      const end = index + marker.length
      if (marker === '<parameter=') {
        this.inParameter = true
        this.header = { type: 'parameter', name: '' }
      } else if (marker === '</parameter>') this.inParameter = false
      else if (this.inParameter) continue
      else if (marker === '<function=') {
        this.inFunction = true
        this.header = { type: 'function', name: '' }
      } else if (marker === '</function>') this.inFunction = false
      yield { marker, index, end }
      index = end - 1
    }
  }
}
