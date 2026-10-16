import { describe, expect, it } from 'vitest'
import {
  AudioBlock,
  Message,
  TextBlock,
  ToolUseBlock,
  ToolResultBlock,
  JsonBlock,
  ReasoningBlock,
  type ToolSpec,
} from '@strands-agents/sdk'
import { toolPrompt } from '../src/gemma/prompt.js'

const tools: ToolSpec[] = [
  {
    name: 'lookup',
    description: 'Look up a value.',
    inputSchema: {
      type: 'object',
      properties: {
        items: {
          type: 'array',
          items: { type: 'object', properties: { label: { type: ['string', 'null'] } } },
        },
        mode: { type: 'string', enum: ['fast', 'slow'] },
        fixed: { const: { type: 'object' } },
      },
      required: ['items'],
      additionalProperties: false,
      $defs: { choice: { oneOf: [{ type: 'number' }, { type: 'boolean' }] } },
    },
  },
]
const user = (text: string) => new Message({ role: 'user', content: [new TextBlock(text)] })
const call = new Message({
  role: 'assistant',
  content: [new ToolUseBlock({ name: 'lookup', toolUseId: 'a', input: { key: 'x' } })],
})
const result = new Message({
  role: 'user',
  content: [
    new ToolResultBlock({
      toolUseId: 'a',
      status: 'success',
      content: [new JsonBlock({ json: { answer: 42 } })],
    }),
  ],
})

describe('Gemma tool prompts', () => {
  it('preserves all thoughts in a tool loop, including when thinking is later disabled', () => {
    const thoughtCall = new Message({
      role: 'assistant',
      content: [new ReasoningBlock({ text: 'Find x.' }), ...call.content],
    })
    const second = new Message({
      role: 'assistant',
      content: [new ReasoningBlock({ text: 'Check again.' }), ...call.content],
    })
    const history = [user('Find'), thoughtCall, result, second, result]
    const snapshot = JSON.stringify(history)
    const prompt = toolPrompt(history, tools, {}, true)
    expect(prompt).toContain('<|turn>system\n<|think|>\n')
    expect(prompt).toContain('<|channel>thought\nFind x.<channel|>')
    expect(prompt).toContain('<tool_response|><|channel>thought\nCheck again.<channel|>')
    expect(toolPrompt(history, tools, {}, false)).toContain('<|channel>thought\nFind x.<channel|>')
    const injected = new Message({
      role: 'user',
      content: [...result.content, new TextBlock('New input')],
    })
    expect(toolPrompt([history[0]!, thoughtCall, injected], tools, {}, true)).not.toContain(
      'Find x.',
    )
    expect(JSON.stringify(history)).toBe(snapshot)
  })

  it.each([{ signature: 'signed' }, { redactedContent: new Uint8Array([1]) }])(
    'rejects reasoning that Gemma cannot carry: %s',
    (value) => {
      const message = new Message({ role: 'assistant', content: [new ReasoningBlock(value)] })
      expect(() => toolPrompt([message], tools, {})).toThrow('signed or redacted')
    },
  )
  it('uses native declarations, preserves the system prompt and keeps schema data unchanged', () => {
    const prompt = toolPrompt([user('Find x')], tools, { systemPrompt: 'Be brief.' })
    expect(prompt.startsWith('<|turn>system\nBe brief.<|tool>declaration:lookup')).toBe(true)
    expect(prompt).toContain('type:<|"|>OBJECT<|"|>')
    expect(prompt).toContain('type:[<|"|>STRING<|"|>,<|"|>NULL<|"|>]')
    expect(prompt).toContain('enum:[<|"|>fast<|"|>,<|"|>slow<|"|>]')
    expect(prompt).toContain('const:{type:<|"|>object<|"|>}')
    expect(prompt.endsWith('<|turn>user\nFind x<turn|>\n<|turn>model\n')).toBe(true)
    expect(prompt).not.toContain('<bos>')
  })

  it('keeps results and the next generation inside the same model turn', () => {
    const prompt = toolPrompt([user('Find x'), call, result], tools, {})
    expect(prompt).toContain(
      '<|turn>model\n<|tool_call>call:lookup{key:<|"|>x<|"|>}<tool_call|><|tool_response>response:lookup',
    )
    expect(prompt.endsWith('{value:<|"|>{"answer":42}<|"|>}<tool_response|>')).toBe(true)
    expect(prompt.match(/<\|turn>model/g)).toHaveLength(1)
  })

  it('closes old turns, continues after tools and preserves later user input', () => {
    const final = new Message({ role: 'assistant', content: [new TextBlock('42')] })
    const prompt = toolPrompt([user('Find x'), call, result, final, user('Thanks')], tools, {})
    expect(prompt).toContain('<tool_response|>42<turn|>\n<|turn>user\nThanks')
    expect(prompt.endsWith('<|turn>model\n')).toBe(true)
    const injected = new Message({
      role: 'user',
      content: [...result.content, new TextBlock('Use it.')],
    })
    expect(toolPrompt([user('Find x'), call, injected], tools, {})).toContain(
      '<tool_response|><turn|>\n<|turn>user\nUse it.',
    )
  })

  it('supports tools without schemas and passes tool failures back to the model', () => {
    const failed = new Message({
      role: 'user',
      content: [
        new ToolResultBlock({
          toolUseId: 'a',
          status: 'error',
          content: [new TextBlock('Missing key')],
        }),
      ],
    })
    const prompt = toolPrompt(
      [user('Find x'), call, failed],
      [{ name: 'lookup', description: 'Lookup' }],
      {},
    )
    expect(prompt).toContain('parameters:{type:<|"|>OBJECT<|"|>,properties:{}}')
    expect(prompt).toContain('Tool error: Missing key')
  })

  it('rejects missing, unmatched, and duplicate tool results', () => {
    expect(() => toolPrompt([result], tools, {})).toThrow('matching call')
    expect(() => toolPrompt([call], tools, {})).toThrow('missing a tool result')
    expect(() => toolPrompt([call, user('Next')], tools, {})).toThrow('missing a tool result')
    expect(() => toolPrompt([call, call], tools, {})).toThrow('missing a tool result')
    const duplicate = new Message({
      role: 'assistant',
      content: [...call.content, ...call.content],
    })
    expect(() => toolPrompt([duplicate], tools, {})).toThrow('duplicate tool call ID')
  })

  it('rejects media instead of turning it into a text-only placeholder', () => {
    const audio = new Message({
      role: 'user',
      content: [new AudioBlock({ format: 'wav', source: { bytes: new Uint8Array([1]) } })],
    })
    expect(() => toolPrompt([audio], tools, {})).toThrow('text and tool results only')
  })
})
