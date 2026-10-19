import { describe, expect, it, vi } from 'vitest'
import {
  AudioBlock,
  ImageBlock,
  JsonBlock,
  Message,
  ReasoningBlock,
  TextBlock,
  ToolResultBlock,
  ToolUseBlock,
} from '@strands-agents/sdk'
import { ModelType, postInitAndCheckFieldsChatCompletion, prebuiltAppConfig } from '@mlc-ai/web-llm'
import { formatMessages } from '../../src/chat/messages.js'
import { WebLLMModel } from '../../src/index.js'
import { chunk, collect, setup } from '../../fixtures/engine.js'

describe('messages', () => {
  it('omits an empty system prompt so Hermes can supply its tool prompt', () => {
    const messages = [new Message({ role: 'user', content: [new TextBlock('Use the clock.')] })]
    for (const prompt of ['', [], [new TextBlock('')]]) {
      const output = formatMessages(messages, prompt)
      expect(output).toEqual([{ role: 'user', content: 'Use the clock.' }])
    }
  })

  it('joins text blocks into the one text field WebLLM accepts', () => {
    const messages = [
      new Message({ role: 'user', content: [new TextBlock('One'), new TextBlock('Two')] }),
    ]
    const output = formatMessages(messages, 'Be brief')
    expect(output).toEqual([
      { role: 'system', content: 'Be brief' },
      { role: 'user', content: 'One\nTwo' },
    ])
    expect(() =>
      postInitAndCheckFieldsChatCompletion({ messages: output }, 'test-model', ModelType.LLM),
    ).not.toThrow()
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
    expect(output).toEqual([{ role: 'tool', tool_call_id: 'a', content: 'Tool error: Not found' }])
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

  it('adds a non-empty tool reply when the result contains only an image', () => {
    const output = formatMessages([
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
      { role: 'tool', tool_call_id: 'a', content: 'Tool successfully returned an image.' },
      {
        role: 'user',
        content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,AQID' } }],
      },
    ])
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

  it('omits an empty assistant turn with no tool calls', () => {
    const output = formatMessages([
      new Message({ role: 'assistant', content: [new ReasoningBlock({ text: '' })] }),
      new Message({ role: 'user', content: [new TextBlock('Continue.')] }),
    ])

    expect(output).toEqual([{ role: 'user', content: 'Continue.' }])
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

  it('passes WAV bytes for models with audio support', () => {
    const output = formatMessages([
      new Message({
        role: 'user',
        content: [new AudioBlock({ format: 'wav', source: { bytes: new Uint8Array([1, 2, 3]) } })],
      }),
    ])
    expect(output[0]).toEqual({
      role: 'user',
      content: [
        { type: 'text', text: '' },
        { type: 'input_audio', input_audio: { format: 'wav', data: 'AQID' } },
      ],
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
        { type: 'text', text: '' },
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
