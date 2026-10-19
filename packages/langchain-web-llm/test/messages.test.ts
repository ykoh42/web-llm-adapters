import { describe, expect, it } from 'vitest'
import {
  AIMessage,
  ChatMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from '@langchain/core/messages'
import { ModelType, postInitAndCheckFieldsChatCompletion } from '@mlc-ai/web-llm'
import { formatMessages, nativeMessages } from '../src/chat/messages.js'
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
  it("combines adjacent text into WebLLM's single text part", () => {
    const output = formatMessages([
      new HumanMessage({
        content: [
          { type: 'text', text: 'Describe this. ' },
          { type: 'text', text: 'Focus on the label.' },
          { type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } },
        ],
      }),
    ])

    expect(output).toEqual([
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Describe this. Focus on the label.' },
          { type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } },
        ],
      },
    ])
    expect(() =>
      postInitAndCheckFieldsChatCompletion({ messages: output }, 'test-model', ModelType.VLM),
    ).not.toThrow()
  })
  it('rejects text blocks split by media when merging would change their order', () => {
    expect(() =>
      formatMessages([
        new HumanMessage({
          content: [
            { type: 'text', text: 'Before. ' },
            { type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } },
            { type: 'text', text: 'After.' },
          ],
        }),
      ]),
    ).toThrow('cannot preserve text around media blocks')
  })
  it('preserves image-first order when the user content is valid', () => {
    const output = formatMessages([
      new HumanMessage({
        content: [
          { type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } },
          { type: 'text', text: 'Describe this.' },
        ],
      }),
    ])

    expect(output[0]?.content).toEqual([
      { type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } },
      { type: 'text', text: 'Describe this.' },
    ])
    expect(() =>
      postInitAndCheckFieldsChatCompletion({ messages: output }, 'test-model', ModelType.VLM),
    ).not.toThrow()
  })
  it('merges adjacent text while keeping surrounding images in order', () => {
    const output = formatMessages([
      new HumanMessage({
        content: [
          { type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } },
          { type: 'text', text: 'First. ' },
          { type: 'text', text: 'Second.' },
          { type: 'image_url', image_url: { url: 'data:image/jpeg;base64,AQ==' } },
        ],
      }),
    ])

    expect(output[0]?.content).toEqual([
      { type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } },
      { type: 'text', text: 'First. Second.' },
      { type: 'image_url', image_url: { url: 'data:image/jpeg;base64,AQ==' } },
    ])
    expect(() =>
      postInitAndCheckFieldsChatCompletion({ messages: output }, 'test-model', ModelType.VLM),
    ).not.toThrow()
  })
  it('rejects image detail before WebLLM receives the request', () => {
    const input = [
      {
        role: 'user' as const,
        content: [
          {
            type: 'image_url' as const,
            image_url: { url: 'data:image/png;base64,AA==', detail: 'high' as const },
          },
        ],
      },
    ]

    expect(() =>
      formatMessages([
        new HumanMessage({
          content: [
            {
              type: 'image_url',
              image_url: { url: 'data:image/png;base64,AA==', detail: 'high' },
            },
          ],
        }),
      ]),
    ).toThrow('WebLLM does not support image_url.detail')
    expect(() =>
      postInitAndCheckFieldsChatCompletion({ messages: input }, 'test-model', ModelType.VLM),
    ).toThrow('detail')
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
          content: [{ type: 'video', url: 'x' }],
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
  it('preserves supported names and validates names represented by tool call IDs', () => {
    const history = formatMessages([
      new HumanMessage({ content: 'Hi', name: 'Ada' }),
      new AIMessage({ content: 'Hello', name: 'Assistant' }),
    ])

    expect(nativeMessages(history)).toEqual([
      { role: 'user', content: 'Hi', name: 'Ada' },
      { role: 'assistant', content: 'Hello', name: 'Assistant' },
    ])
    expect(() => formatMessages([new SystemMessage({ content: 'Rules', name: 'System' })])).toThrow(
      'system messages cannot preserve names',
    )
    expect(() =>
      formatMessages([new ToolMessage({ content: 'Done', tool_call_id: 'call', name: 'Tool' })]),
    ).toThrow('matching assistant tool call')
    expect(
      formatMessages([
        new AIMessage({ content: '', tool_calls: [{ id: 'call', name: 'lookup', args: {} }] }),
        new ToolMessage({ content: 'Done', tool_call_id: 'call', name: 'lookup' }),
      ]),
    ).toMatchObject([
      { role: 'assistant', tool_calls: [{ id: 'call', function: { name: 'lookup' } }] },
      { role: 'tool', tool_call_id: 'call', content: 'Done' },
    ])
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

it('hoists images after every tool result without changing IDs, text, or history', () => {
  const history = [
    new HumanMessage('Compare the pictures.'),
    new AIMessage({
      content: '',
      tool_calls: [
        { id: 'one', name: 'picture', args: {} },
        { id: 'two', name: 'picture', args: {} },
      ],
    }),
    new ToolMessage({
      tool_call_id: 'one',
      content: [
        { type: 'text', text: 'First picture.' },
        { type: 'image_url', image_url: { url: 'https://example.test/one.png' } },
      ],
    }),
    new ToolMessage({
      tool_call_id: 'two',
      content: [{ type: 'image', data: 'AA==', mimeType: 'image/png' }],
    }),
    new HumanMessage('Which is brighter?'),
  ]
  const original = history.map((message) => message.toDict())
  const output = formatMessages(history)

  expect(output.map((message) => message.role)).toEqual([
    'user',
    'assistant',
    'tool',
    'tool',
    'user',
    'user',
  ])
  expect(output[1]?.content).toBe('')
  expect(output.slice(2, 5)).toEqual([
    { role: 'tool', tool_call_id: 'one', content: 'First picture.' },
    { role: 'tool', tool_call_id: 'two', content: '[Image attached]' },
    {
      role: 'user',
      content: [
        { type: 'image_url', image_url: { url: 'https://example.test/one.png' } },
        { type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } },
      ],
    },
  ])
  expect(history.map((message) => message.toDict())).toEqual(original)
  expect(formatMessages(history.slice(0, -1)).at(-1)).toEqual(output[4])
})

it('passes tool images to native vision input and preserves Gemma protocol limits', async () => {
  const { engine, create } = setup(chunk({ content: 'A picture.' }, 'stop'))
  const history = [
    new HumanMessage('Describe the picture.'),
    new AIMessage({ content: '', tool_calls: [{ id: 'picture', name: 'read', args: {} }] }),
    new ToolMessage({
      tool_call_id: 'picture',
      content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } }],
    }),
  ]
  const model = new ChatWebLLM({ engine, modelId: 'test-vision-model' })

  expect((await model.invoke(history)).text).toBe('A picture.')
  expect(create.mock.calls[0]?.[0].messages.at(-2)).toEqual({
    role: 'tool',
    tool_call_id: 'picture',
    content: '[Image attached]',
  })
  expect(create.mock.calls[0]?.[0].messages.at(-1)).toEqual({
    role: 'user',
    content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } }],
  })
  const gemma = new ChatWebLLM({ engine, modelId: 'gemma-4-E2B-it-q4f16_1-MLC' })
  await expect(gemma.invoke(history)).rejects.toThrow('text and tool results only')
  expect(create).toHaveBeenCalledTimes(1)
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

it('encodes LangChain image and WAV byte blocks without losing view offsets', () => {
  const buffer = Uint8Array.from({ length: 24_580 }, (_, index) => index % 256)
  const bytes = buffer.subarray(1, 24_579)
  const data = Buffer.from(bytes).toString('base64')
  const output = formatMessages([
    new HumanMessage({
      content: [
        { type: 'image', source_type: 'base64', data: bytes, mime_type: 'image/png' },
        { type: 'audio', source_type: 'base64', data: bytes, mime_type: 'audio/wav' },
      ],
    }),
  ])

  expect(output[0]?.content).toEqual([
    { type: 'image_url', image_url: { url: `data:image/png;base64,${data}` } },
    { type: 'input_audio', input_audio: { format: 'wav', data } },
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
  expect(new ChatWebLLM({ engine, modelId: 'test' }).profile.imageToolMessage).toBe(false)
  expect(
    new ChatWebLLM({
      engine,
      modelId: 'custom-vision-tools',
      profile: { imageInputs: true, imageToolMessage: true },
    }).profile,
  ).toMatchObject({ imageInputs: true, imageToolMessage: true })
  expect(
    new ChatWebLLM({ engine, modelId: 'Phi-3.5-vision-instruct-q4f16_1-MLC' }).profile.imageInputs,
  ).toBe(true)
  expect(
    new ChatWebLLM({ engine, modelId: 'Phi-3.5-vision-instruct-q4f16_1-MLC' }).profile
      .imageToolMessage,
  ).toBe(false)
})

it.each([0, -1, NaN, 1.5])('rejects invalid model context limits %s', (maxInputTokens) => {
  const { engine } = setup()
  expect(() => new ChatWebLLM({ engine, modelId: 'test', profile: { maxInputTokens } })).toThrow(
    'positive integers',
  )
})
