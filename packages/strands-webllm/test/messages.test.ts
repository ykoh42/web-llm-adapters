import { describe, expect, it } from 'vitest'
import {
  AudioBlock,
  ImageBlock,
  JsonBlock,
  Message,
  TextBlock,
  ToolResultBlock,
} from '@strands-agents/sdk'
import { ModelType, postInitAndCheckFieldsChatCompletion, prebuiltAppConfig } from '@mlc-ai/web-llm'
import { formatMessages } from '../src/messages.js'
import { WebLLMModel } from '../src/index.js'
import { chunk, collect, setup } from './helpers.js'

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
