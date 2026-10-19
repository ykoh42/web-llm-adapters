import { describe, expect, it } from 'vitest'
import {
  AIMessage,
  ChatMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from '@langchain/core/messages'
import { formatMessages } from '../src/chat/messages.js'
import { ChatWebLLM } from '../src/index.js'
import { chunk, setup } from '../fixtures/engine.js'

describe('message and option boundaries', () => {
  it.each([
    { type: 'image', url: 'https://example.test/pic.png' },
    { type: 'image', data: 'AA==', mimeType: 'image/png' },
    { type: 'image', base64: 'AA==', mime_type: 'image/png' },
    { type: 'image_url', image_url: 'https://example.test/pic.png' },
    { type: 'audio', data: 'AA==', mimeType: 'audio/wav' },
  ])('maps standard media blocks $type', (block) => {
    const result = formatMessages([new HumanMessage({ content: [block] })])
    expect(result[0]?.role).toBe('user')
    expect(result[0]?.content).toEqual([
      block.type === 'audio'
        ? { type: 'input_audio', input_audio: { format: 'wav', data: 'AA==' } }
        : {
            type: 'image_url',
            image_url: {
              url:
                'url' in block
                  ? block.url
                  : 'image_url' in block
                    ? block.image_url
                    : 'data:image/png;base64,AA==',
            },
          },
    ])
  })
  it('rejects missing IDs, invalid calls, late system messages and unsupported roles', () => {
    expect(() =>
      formatMessages([new AIMessage({ content: '', tool_calls: [{ name: 'missing', args: {} }] })]),
    ).toThrow('ID')
    expect(() =>
      formatMessages([
        new AIMessage({
          content: '',
          invalid_tool_calls: [{ name: 'bad', args: '{', error: 'invalid' }],
        }),
      ]),
    ).toThrow('invalid tool')
    expect(() => formatMessages([new HumanMessage('Hi'), new SystemMessage('Late')])).toThrow(
      'before',
    )
    expect(() => formatMessages([new ChatMessage('Hi', 'custom')])).toThrow('generic')
    expect(() =>
      formatMessages([
        new ToolMessage({
          content: [{ type: 'image_url', image_url: { url: 'x' } }],
          tool_call_id: 'id',
        }),
      ]),
    ).toThrow('cannot use')
  })
  it('reads compatible reasoning_content and rejects signed or non-text reasoning', () => {
    expect(
      formatMessages([
        new AIMessage({ content: 'Answer', additional_kwargs: { reasoning_content: 'Thought' } }),
      ])[0],
    ).toMatchObject({ reasoning: 'Thought' })
    expect(() =>
      formatMessages([
        new AIMessage({
          content: [{ type: 'reasoning', reasoning: 'Thought', signature: 'signed' }],
        }),
      ]),
    ).toThrow('unsigned')
    expect(() =>
      formatMessages([
        new AIMessage({
          content: 'Answer',
          additional_kwargs: { reasoning_content: { value: 'Thought' } },
        }),
      ]),
    ).toThrow('text reasoning')
  })
  it('validates names and maps constructor settings to native requests', async () => {
    const { engine, create } = setup(chunk({ content: 'Hi' }, 'stop'))
    expect(() => new ChatWebLLM({ engine, modelId: '' })).toThrow('modelId')
    const model = new ChatWebLLM({
      engine,
      modelId: 'test-model',
      temperature: 0.2,
      maxTokens: 100,
      topP: 0.8,
      stop: ['END'],
    })
    await model.invoke('Hi')
    expect(create.mock.calls[0]?.[0]).toMatchObject({
      temperature: 0.2,
      max_tokens: 100,
      top_p: 0.8,
      stop: ['END'],
    })
    expect(
      model.invocationParams({ request: { temperature: 0.3 }, stop: ['FINISH'] }),
    ).toMatchObject({ model: 'test-model', temperature: 0.3, stop: ['FINISH'] })
    expect(model._llmType()).toBe('webllm')
    expect(model._identifyingParams()).not.toHaveProperty('engine')
  })
})

it('accepts LangChain standard base64 image and WAV blocks', () => {
  const output = formatMessages([
    new HumanMessage({
      content: [
        { type: 'image', source_type: 'base64', data: 'AA==', mime_type: 'image/png' },
        { type: 'audio', source_type: 'base64', data: 'AQ==', mime_type: 'audio/wav' },
      ],
    }),
  ])
  expect(output[0]?.content).toEqual([
    { type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } },
    { type: 'input_audio', input_audio: { format: 'wav', data: 'AQ==' } },
  ])
})

it('normalizes plain text blocks and provider tool blocks without duplicating calls', () => {
  const history = formatMessages([
    new HumanMessage({
      content: [
        { type: 'text', text: 'First' },
        { type: 'text', text: ' last' },
      ],
    }),
    new AIMessage({
      content: [
        { type: 'text', text: 'Call' },
        { type: 'tool_use', id: 'one', name: 'lookup', input: { code: 'A7' } },
      ],
      tool_calls: [{ id: 'one', name: 'lookup', args: { code: 'A7' } }],
    }),
  ])
  expect(history[0]?.content).toBe('First last')
  expect(history[1]).toMatchObject({
    content: 'Call',
    tool_calls: [{ id: 'one', function: { name: 'lookup', arguments: '{"code":"A7"}' } }],
  })
  expect(history[1]?.role === 'assistant' && history[1].tool_calls).toHaveLength(1)
  expect(() =>
    formatMessages([
      new AIMessage({
        content: [{ type: 'tool_use', id: 'one', name: 'lookup', input: { code: 'wrong' } }],
        tool_calls: [{ id: 'one', name: 'lookup', args: { code: 'A7' } }],
      }),
    ]),
  ).toThrow('conflicting')
})

it('exposes model capabilities and caller-owned context limits without guessing them', () => {
  const { engine } = setup()
  const model = new ChatWebLLM({
    engine,
    modelId: 'Qwen3.5-4B-q4f16_1-MLC',
    profile: { maxInputTokens: 32768 },
  })
  expect(model.profile).toMatchObject({
    toolCalling: true,
    reasoningOutput: true,
    maxInputTokens: 32768,
    pdfInputs: false,
  })
  const profile = model.profile
  profile.maxInputTokens = 1
  expect(model.profile.maxInputTokens).toBe(32768)
  expect(new ChatWebLLM({ engine, modelId: 'test' }).profile.maxInputTokens).toBeUndefined()
  expect(new ChatWebLLM({ engine, modelId: 'Qwen3-1.7B-q4f16_1-MLC' }).profile.toolCalling).toBe(
    false,
  )
  expect(
    new ChatWebLLM({ engine, modelId: 'Phi-3.5-vision-instruct-q4f16_1-MLC' }).profile.imageInputs,
  ).toBe(true)
})

it.each([0, -1, NaN, 1.5])('rejects invalid model context limits %s', (maxInputTokens) => {
  const { engine } = setup()
  expect(() => new ChatWebLLM({ engine, modelId: 'test', profile: { maxInputTokens } })).toThrow(
    'positive integers',
  )
})
