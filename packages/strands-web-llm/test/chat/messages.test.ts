import { runInNewContext } from 'node:vm'
import { describe, expect, it, vi } from 'vitest'
import {
  AudioBlock,
  CachePointBlock,
  CitationsBlock,
  ImageBlock,
  JsonBlock,
  Message,
  ReasoningBlock,
  TextBlock,
  ToolResultBlock,
  ToolUseBlock,
} from '@strands-agents/sdk'
import { ModelType, postInitAndCheckFieldsChatCompletion, prebuiltAppConfig } from '@mlc-ai/web-llm'
import { formatMessages, lastQueryIndex } from '../../src/chat/messages.js'
import { WebLLMModel } from '../../src/index.js'
import { chunk, collect, setup } from '../../fixtures/engine.js'

describe('last user query', () => {
  const question = new Message({ role: 'user', content: [new TextBlock('Find the code.')] })
  const answer = new Message({ role: 'assistant', content: [new TextBlock('Checking.')] })
  const result = new ToolResultBlock({
    toolUseId: 'lookup',
    status: 'success',
    content: [new TextBlock('42')],
  })
  const reply = new Message({ role: 'user', content: [result] })
  const hints = new Message({
    role: 'user',
    content: [result, new CachePointBlock({ cacheType: 'default' })],
  })
  const context = new Message({
    role: 'user',
    content: [result, new TextBlock('Current task context.')],
  })

  it.each([
    { name: 'empty history', messages: [], trailing: 0, index: -1 },
    { name: 'assistant only', messages: [answer], trailing: 0, index: -1 },
    { name: 'tool reply only', messages: [reply], trailing: 0, index: -1 },
    { name: 'tool loop', messages: [question, answer, reply], trailing: 0, index: 0 },
    { name: 'tool cache hints', messages: [question, answer, hints], trailing: 1, index: 0 },
    { name: 'injected context', messages: [question, answer, context], trailing: 1, index: 0 },
    {
      name: 'user text after a result',
      messages: [question, answer, context],
      trailing: 0,
      index: 2,
    },
    { name: 'new question', messages: [question, answer, reply, question], trailing: 1, index: 3 },
  ])(
    'finds the query in $name without changing frozen history',
    ({ messages, trailing, index }) => {
      const original = JSON.stringify(messages)
      for (const message of messages) {
        for (const block of message.content) Object.freeze(block)
        Object.freeze(message.content)
        Object.freeze(message)
      }
      Object.freeze(messages)

      expect(lastQueryIndex(messages, trailing)).toBe(index)
      expect(JSON.stringify(messages)).toBe(original)
    },
  )
})

describe('messages', () => {
  it('preserves an explicitly empty system prompt', () => {
    const messages = [new Message({ role: 'user', content: [new TextBlock('Use the clock.')] })]
    for (const prompt of ['', [], [new TextBlock('')]]) {
      const output = formatMessages(messages, prompt)
      expect(output).toEqual([
        { role: 'system', content: '' },
        { role: 'user', content: 'Use the clock.' },
      ])
    }
  })

  it.each([
    'test-model',
    'gemma-4-E2B-it-q4f16_1-MLC',
    'Qwen3-4B-q4f16_1-MLC',
    'Qwen3.5-4B-q4f16_1-MLC',
  ])('%s sends an open schema for native JSON mode', async (modelId) => {
    const { model, create } = setup(chunk({ content: '{}' }, 'stop'))
    const format = { type: 'json_object' as const }
    model.updateConfig({ modelId, request: { response_format: format } })

    await collect(
      model.stream([new Message({ role: 'user', content: [new TextBlock('Return an object.')] })]),
    )

    expect(JSON.parse(create.mock.calls[0]?.[0].response_format?.schema ?? 'null')).toEqual({})
    expect(format).toEqual({ type: 'json_object' })
  })

  it.each(['test-model', 'Qwen3-4B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC'])(
    '%s rejects invalid numeric history before encoding tool arguments',
    async (modelId) => {
      for (const value of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
        const { model, create } = setup(chunk({ content: 'Done' }, 'stop'))
        model.updateConfig({ modelId })
        const history = [
          new Message({ role: 'user', content: [new TextBlock('Look up the value.')] }),
          new Message({
            role: 'assistant',
            content: [
              new ToolUseBlock({
                toolUseId: 'call',
                name: 'lookup',
                input: { values: [{ value }] },
              }),
            ],
          }),
          new Message({
            role: 'user',
            content: [
              new ToolResultBlock({
                toolUseId: 'call',
                status: 'success',
                content: [new TextBlock('Done')],
              }),
            ],
          }),
        ]

        await expect(collect(model.stream(history))).rejects.toThrow('finite numbers')
        expect(create).not.toHaveBeenCalled()
      }
    },
  )

  it.each([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY])(
    'rejects invalid JSON tool results instead of encoding them as null: %s',
    async (value) => {
      const { model, create } = setup(chunk({ content: 'Done' }, 'stop'))
      const history = [
        new Message({
          role: 'assistant',
          content: [new ToolUseBlock({ toolUseId: 'call', name: 'lookup', input: {} })],
        }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'call',
              status: 'success',
              content: [new JsonBlock({ json: { values: [value] } })],
            }),
          ],
        }),
      ]

      await expect(collect(model.stream(history))).rejects.toThrow('finite numbers')
      expect(create).not.toHaveBeenCalled()
    },
  )

  it('keeps finite JSON tool arguments and results byte-for-byte unchanged', async () => {
    const input = {
      tiny: 1e-300,
      large: 1e300,
      integer: Number.MAX_SAFE_INTEGER,
      values: [null, 'NaN', 'Infinity', '-Infinity', '<tool_call>'],
    }
    const { model, create } = setup(chunk({ content: 'Done' }, 'stop'))
    const history = [
      new Message({ role: 'user', content: [new TextBlock('Look up the value.')] }),
      new Message({
        role: 'assistant',
        content: [new ToolUseBlock({ toolUseId: 'call', name: 'lookup', input })],
      }),
      new Message({
        role: 'user',
        content: [
          new ToolResultBlock({
            toolUseId: 'call',
            status: 'success',
            content: [new JsonBlock({ json: input })],
          }),
        ],
      }),
    ]

    await collect(model.stream(history))

    const assistant = create.mock.calls[0]?.[0].messages[1]
    if (assistant?.role !== 'assistant') throw new Error('Missing assistant history.')
    expect(assistant.tool_calls?.[0]?.function.arguments).toBe(JSON.stringify(input))
    expect(create.mock.calls[0]?.[0].messages[2]?.content).toBe(JSON.stringify(input))
  })

  it('joins text blocks into the one text field WebLLM accepts', () => {
    const messages = [
      new Message({ role: 'user', content: [new TextBlock('One'), new TextBlock('Two')] }),
    ]
    const output = formatMessages(messages, 'Be brief')
    expect(output).toEqual([
      { role: 'system', content: 'Be brief' },
      { role: 'user', content: 'OneTwo' },
    ])
    expect(() =>
      postInitAndCheckFieldsChatCompletion({ messages: output }, 'test-model', ModelType.LLM),
    ).not.toThrow()
  })

  it('joins system prompt blocks without adding separators', () => {
    const output = formatMessages([], [new TextBlock('One'), new TextBlock('Two')])

    expect(output).toEqual([{ role: 'system', content: 'OneTwo' }])
  })

  it('puts tool results before injected user text', () => {
    const output = formatMessages([
      new Message({
        role: 'user',
        content: [
          new TextBlock('Keep going'),
          new ToolResultBlock({
            toolUseId: 'a',
            status: 'success',
            content: [new JsonBlock({ json: { count: 2 } })],
          }),
        ],
      }),
    ])
    expect(output).toEqual([
      { role: 'tool', tool_call_id: 'a', content: '{"count":2}' },
      { role: 'user', content: 'Keep going' },
    ])
  })

  it('keeps tool errors visible to the model', () => {
    const output = formatMessages([
      new Message({
        role: 'user',
        content: [
          new ToolResultBlock({
            toolUseId: 'a',
            status: 'error',
            content: [new TextBlock('Not found')],
          }),
        ],
      }),
    ])
    expect(output).toEqual([{ role: 'tool', tool_call_id: 'a', content: '[ERROR] Not found' }])
  })

  it('preserves whitespace in tool results', () => {
    const output = formatMessages([
      new Message({
        role: 'user',
        content: [
          new ToolResultBlock({
            toolUseId: 'a',
            status: 'success',
            content: [new TextBlock('    return 1\n')],
          }),
        ],
      }),
    ])

    expect(output).toEqual([{ role: 'tool', tool_call_id: 'a', content: '    return 1\n' }])
  })

  it('joins tool-result text blocks without adding separators', () => {
    const output = formatMessages([
      new Message({
        role: 'user',
        content: [
          new ToolResultBlock({
            toolUseId: 'a',
            status: 'success',
            content: [new TextBlock('One'), new TextBlock('Two')],
          }),
        ],
      }),
    ])

    expect(output).toEqual([{ role: 'tool', tool_call_id: 'a', content: 'OneTwo' }])
  })

  it('keeps tool-result images separate and after adjacent tool replies', () => {
    const output = formatMessages([
      new Message({
        role: 'assistant',
        content: [
          new ToolUseBlock({ toolUseId: 'a', name: 'read', input: {} }),
          new ToolUseBlock({ toolUseId: 'b', name: 'read', input: {} }),
        ],
      }),
      new Message({
        role: 'user',
        content: [
          new ToolResultBlock({
            toolUseId: 'a',
            status: 'success',
            content: [
              new TextBlock('Found the page.'),
              new ImageBlock({ format: 'png', source: { url: 'https://example.com/page.png' } }),
            ],
          }),
          new ToolResultBlock({
            toolUseId: 'b',
            status: 'success',
            content: [
              new TextBlock('Done.'),
              new ImageBlock({ format: 'jpeg', source: { url: 'https://example.com/other.jpg' } }),
            ],
          }),
          new TextBlock('What is in the image?'),
        ],
      }),
    ])

    expect(output.slice(-5)).toEqual([
      { role: 'tool', tool_call_id: 'a', content: 'Found the page.' },
      { role: 'tool', tool_call_id: 'b', content: 'Done.' },
      {
        role: 'user',
        content: [{ type: 'image_url', image_url: { url: 'https://example.com/page.png' } }],
      },
      {
        role: 'user',
        content: [{ type: 'image_url', image_url: { url: 'https://example.com/other.jpg' } }],
      },
      { role: 'user', content: 'What is in the image?' },
    ])
    expect(() =>
      postInitAndCheckFieldsChatCompletion({ messages: output }, 'test-model', ModelType.VLM),
    ).not.toThrow()
  })

  it('keeps image-only tool replies empty and attaches the image separately', () => {
    const output = formatMessages([
      new Message({
        role: 'assistant',
        content: [new ToolUseBlock({ toolUseId: 'a', name: 'read', input: {} })],
      }),
      new Message({
        role: 'user',
        content: [
          new ToolResultBlock({
            toolUseId: 'a',
            status: 'success',
            content: [
              new ImageBlock({ format: 'png', source: { bytes: new Uint8Array([1, 2, 3]) } }),
            ],
          }),
        ],
      }),
    ])

    expect(output).toEqual([
      {
        role: 'assistant',
        content: '',
        tool_calls: [{ id: 'a', type: 'function', function: { name: 'read', arguments: '{}' } }],
      },
      { role: 'tool', tool_call_id: 'a', content: '' },
      {
        role: 'user',
        content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,AQID' } }],
      },
    ])
    expect(() =>
      postInitAndCheckFieldsChatCompletion({ messages: output }, 'test-model', ModelType.VLM),
    ).not.toThrow()
  })

  it('rejects text on both sides of media when WebLLM cannot preserve its order', () => {
    const content = [
      new TextBlock('Before'),
      new ImageBlock({ format: 'png', source: { url: 'https://example.com/image.png' } }),
      new TextBlock('After'),
    ]
    const messages = [new Message({ role: 'user', content })]

    expect(() => formatMessages(messages)).toThrow(
      'WebLLM accepts one text part per user message and cannot preserve text around media blocks.',
    )
  })

  it('keeps empty assistant tool-call content as a string WebLLM accepts', () => {
    const output = formatMessages([
      new Message({ role: 'user', content: [new TextBlock('Get the time.')] }),
      new Message({
        role: 'assistant',
        content: [new ToolUseBlock({ toolUseId: 'a', name: 'clock', input: {} })],
      }),
      new Message({
        role: 'user',
        content: [
          new ToolResultBlock({
            toolUseId: 'a',
            status: 'success',
            content: [new TextBlock('12:00')],
          }),
        ],
      }),
    ])
    expect(output).toEqual([
      { role: 'user', content: 'Get the time.' },
      {
        role: 'assistant',
        content: '',
        tool_calls: [{ id: 'a', type: 'function', function: { name: 'clock', arguments: '{}' } }],
      },
      { role: 'tool', tool_call_id: 'a', content: '12:00' },
    ])
    expect(() =>
      postInitAndCheckFieldsChatCompletion(
        { messages: output },
        'Hermes-3-Llama-3.1-8B-q4f16_1-MLC',
        ModelType.LLM,
      ),
    ).not.toThrow()
  })

  it('rejects tool signatures that WebLLM cannot preserve', () => {
    const message = new Message({
      role: 'assistant',
      content: [
        new ToolUseBlock({
          toolUseId: 'call-1',
          name: 'lookup',
          input: { city: 'Seoul' },
          reasoningSignature: 'opaque-signature',
        }),
      ],
    })

    expect(() => formatMessages([message])).toThrow('tool calls with reasoning signatures')
  })

  it('preserves empty user turns WebLLM accepts', () => {
    for (const content of [[], [new CachePointBlock({ cacheType: 'default' })]]) {
      const output = formatMessages([new Message({ role: 'user', content })])
      expect(output).toEqual([{ role: 'user', content: '' }])
      expect(() =>
        postInitAndCheckFieldsChatCompletion({ messages: output }, 'test-model', ModelType.LLM),
      ).not.toThrow()
    }
  })

  it('preserves empty assistant turns WebLLM accepts', () => {
    for (const content of [[], [new TextBlock('')], [new ReasoningBlock({ text: '' })]]) {
      const output = formatMessages([
        new Message({ role: 'assistant', content }),
        new Message({ role: 'user', content: [new TextBlock('Continue.')] }),
      ])

      expect(output).toEqual([
        { role: 'assistant', content: '' },
        { role: 'user', content: 'Continue.' },
      ])
      expect(() =>
        postInitAndCheckFieldsChatCompletion({ messages: output }, 'test-model', ModelType.LLM),
      ).not.toThrow()
    }
  })

  it('passes image URLs and bytes in WebLLM format', () => {
    const output = formatMessages([
      new Message({
        role: 'user',
        content: [
          new TextBlock('What is this?'),
          new ImageBlock({ format: 'png', source: { bytes: new Uint8Array([1, 2, 3]) } }),
          new ImageBlock({ format: 'jpeg', source: { url: 'https://example.com/image.jpg' } }),
        ],
      }),
    ])
    expect(output[0]).toEqual({
      role: 'user',
      content: [
        { type: 'text', text: 'What is this?' },
        { type: 'image_url', image_url: { url: 'data:image/png;base64,AQID' } },
        { type: 'image_url', image_url: { url: 'https://example.com/image.jpg' } },
      ],
    })
  })

  it('rejects transferred image bytes before creating a WebLLM request', async () => {
    const bytes = new Uint8Array([1, 2, 3])
    const message = new Message({
      role: 'user',
      content: [new ImageBlock({ format: 'png', source: { bytes } })],
    })
    structuredClone(bytes, { transfer: [bytes.buffer] })
    const { model, create } = setup(chunk({ content: 'Okay' }, 'stop'))

    await expect(collect(model.stream([message]))).rejects.toThrow(TypeError)
    expect(create).not.toHaveBeenCalled()
  })

  it('passes image and WAV bytes from another realm', () => {
    const bytes = runInNewContext('new Uint8Array([1, 2, 3])') as Uint8Array
    const output = formatMessages([
      new Message({
        role: 'user',
        content: [
          new ImageBlock({ format: 'png', source: { bytes } }),
          new AudioBlock({ format: 'wav', source: { bytes } }),
        ],
      }),
    ])

    expect(output[0]?.content).toEqual([
      { type: 'image_url', image_url: { url: 'data:image/png;base64,AQID' } },
      { type: 'input_audio', input_audio: { format: 'wav', data: 'AQID' } },
    ])
  })

  it('passes WAV bytes for models with audio support', () => {
    const output = formatMessages([
      new Message({
        role: 'user',
        content: [new AudioBlock({ format: 'wav', source: { bytes: new Uint8Array([1, 2, 3]) } })],
      }),
    ])
    expect(output[0]).toEqual({
      role: 'user',
      content: [{ type: 'input_audio', input_audio: { format: 'wav', data: 'AQID' } }],
    })
  })

  it.each([
    { size: 0, browser: false },
    { size: 257, browser: false },
    { size: 131075, browser: false },
    { size: 131075, browser: true },
  ])('preserves media bytes and view offsets: %j', ({ size, browser }) => {
    const buffer = Uint8Array.from({ length: size + 14 }, (_, index) => index % 256)
    const bytes = buffer.subarray(7, size + 7)
    const expected = Buffer.from(bytes).toString('base64')
    if (browser) vi.stubGlobal('Buffer', undefined)
    try {
      const output = formatMessages([
        new Message({
          role: 'user',
          content: [
            new ImageBlock({ format: 'jpg', source: { bytes } }),
            new AudioBlock({ format: 'wav', source: { bytes } }),
          ],
        }),
      ])
      expect(output[0]?.content).toEqual([
        { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${expected}` } },
        { type: 'input_audio', input_audio: { format: 'wav', data: expected } },
      ])
    } finally {
      if (browser) vi.unstubAllGlobals()
    }
  })

  it('rejects unsupported audio instead of silently dropping it', () => {
    expect(() =>
      formatMessages([
        new Message({
          role: 'user',
          content: [new AudioBlock({ format: 'mp3', source: { bytes: new Uint8Array([1]) } })],
        }),
      ]),
    ).toThrow('audioBlock')
  })

  it('rejects citation blocks instead of dropping cited content', () => {
    expect(() =>
      formatMessages([
        new Message({
          role: 'assistant',
          content: [
            new CitationsBlock({
              citations: [
                {
                  location: { type: 'documentChar', documentIndex: 0, start: 0, end: 6 },
                  source: 'doc-0',
                  sourceContent: [{ text: 'source' }],
                  title: 'Document',
                },
              ],
              content: [{ text: 'Cited answer' }],
            }),
          ],
        }),
      ]),
    ).toThrow('citationsBlock')
  })

  it('keeps Gemma tool support limited to what WebLLM actually accepts', () => {
    const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
    expect(prebuiltAppConfig.model_list.some((model) => model.model_id === modelId)).toBe(true)
    expect(() =>
      postInitAndCheckFieldsChatCompletion(
        {
          messages: [{ role: 'user', content: 'Weather?' }],
          tools: [{ type: 'function', function: { name: 'weather' } }],
        },
        modelId,
        ModelType.LLM,
      ),
    ).toThrow()
  })

  it('passes JSON settings in a request WebLLM accepts', async () => {
    const { engine, create } = setup(chunk({}, 'stop'))
    const model = new WebLLMModel({
      engine,
      modelId: 'gemma-4-E2B-it-q4f16_1-MLC',
      request: {
        response_format: { type: 'json_object', schema: '{"type":"object"}' },
      },
    })
    await collect(model.stream([new Message({ role: 'user', content: [new TextBlock('JSON')] })]))
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        response_format: { type: 'json_object', schema: '{"type":"object"}' },
      }),
    )
    expect(() =>
      postInitAndCheckFieldsChatCompletion(
        structuredClone(create.mock.calls[0]![0]),
        model.modelId!,
        ModelType.LLM,
      ),
    ).not.toThrow()
  })

  it('passes named tool choice in a request WebLLM accepts', async () => {
    const { engine, create } = setup(
      chunk(
        { tool_calls: [{ index: 0, function: { name: 'weather', arguments: '{}' } }] },
        'tool_calls',
      ),
    )
    const model = new WebLLMModel({ engine, modelId: 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC' })
    await collect(
      model.stream([new Message({ role: 'user', content: [new TextBlock('Weather?')] })], {
        toolSpecs: [{ name: 'weather', description: 'Get weather' }],
        toolChoice: { tool: { name: 'weather' } },
      }),
    )
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        tool_choice: { type: 'function', function: { name: 'weather' } },
      }),
    )
    expect(() =>
      postInitAndCheckFieldsChatCompletion(
        structuredClone(create.mock.calls[0]![0]),
        model.modelId!,
        ModelType.LLM,
      ),
    ).not.toThrow()
  })
})
