import { describe, expect, it, vi } from 'vitest'
import {
  AIMessage,
  AIMessageChunk,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from '@langchain/core/messages'
import { tool } from '@langchain/core/tools'
import { ChatPromptTemplate } from '@langchain/core/prompts'
import { createAgent } from 'langchain'
import { createDeepAgent } from 'deepagents'
import { z } from 'zod'
import { ChatWebLLM } from '../src/index.js'
import { chunk, chunks, collect, pause, setup } from '../fixtures/engine.js'

const lookup = tool(({ code }) => `Result ${code}: 42`, {
  name: 'lookup',
  description: 'Look up the answer for a code.',
  schema: z.object({ code: z.string() }),
})
const toolOutput =
  '<tool_call>\n<function=lookup>\n<parameter=code>\nA7\n</parameter>\n</function>\n</tool_call>'
const qwen = 'Qwen3.5-0.8B-q4f16_1-MLC'

function qwenModel(...items: ReturnType<typeof chunk>[]) {
  const result = setup(...items)
  return {
    ...result,
    model: new ChatWebLLM({
      engine: result.engine,
      modelId: qwen,
      request: { extra_body: { enable_thinking: true } },
    }),
  }
}

describe('LangChain model', () => {
  it('returns text, usage, and native metadata without owning the engine', async () => {
    const usage = {
      prompt_tokens: 3,
      completion_tokens: 2,
      total_tokens: 5,
      extra: {
        e2e_latency_s: 1,
        prefill_tokens_per_s: 2,
        decode_tokens_per_s: 3,
        time_to_first_token_s: 0.1,
        time_per_output_token_s: 0.2,
      },
    }
    const { model, engine, create } = setup(chunk({ content: 'Hello' }, 'stop'), {
      ...chunk(),
      usage,
    })
    const unload = vi.spyOn(engine, 'unload')
    const result = await model.invoke('Hi')
    expect(result.text).toBe('Hello')
    expect(result.usage_metadata).toMatchObject({
      input_tokens: 3,
      output_tokens: 2,
      total_tokens: 5,
    })
    expect(result.response_metadata).toMatchObject({ finish_reason: 'stop', webllm: usage.extra })
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({
        messages: [{ role: 'user', content: 'Hi' }],
        n: 1,
        stream: true,
        stream_options: { include_usage: true },
      }),
    )
    expect(unload).not.toHaveBeenCalled()
  })

  it('streams the first token before generation finishes and runs callbacks', async () => {
    const { model, create } = setup()
    const hold = pause()
    create.mockResolvedValueOnce(
      (async function* () {
        yield chunk({ content: 'A' })
        await hold.wait
        yield chunk({ content: 'B' }, 'stop')
      })(),
    )
    const callback = vi.fn()
    const stream = await model.stream('Hi', { callbacks: [{ handleLLMNewToken: callback }] })
    const first = await stream.next()
    expect(first.value?.text).toBe('A')
    hold.resume()
    await collect(stream)
    expect(callback.mock.calls.map(([token]) => token).join('')).toBe('AB')
  })

  it('works in a prompt chain, batch, and message serialization', async () => {
    const { model } = setup(chunk({ content: 'Hello' }, 'stop'))
    const chain = ChatPromptTemplate.fromMessages([
      ['system', 'Be concise'],
      ['human', 'Hi {name}'],
    ]).pipe(model)
    expect((await chain.invoke({ name: 'Ada' })).text).toBe('Hello')
    expect((await model.batch(['One', 'Two'])).map((message) => message.text)).toEqual([
      'Hello',
      'Hello',
    ])
    expect(JSON.stringify(model.toJSON())).not.toContain('engine')
  })

  it('passes native options and per-call settings without mutating them', async () => {
    const { engine, create } = setup(chunk({ content: 'yes' }, 'stop'))
    const request = { temperature: 0.5, top_p: 0.8, extra_body: { enable_thinking: false } }
    const model = new ChatWebLLM({ engine, modelId: 'test-model', request })
    request.temperature = 9
    await model.invoke('Hi', { request: { temperature: 0.2, logprobs: true }, stop: ['END'] })
    expect(create).toHaveBeenCalledWith(
      expect.objectContaining({ temperature: 0.2, top_p: 0.8, logprobs: true, stop: ['END'] }),
    )
    expect(model.getLsParams({})).toMatchObject({
      ls_provider: 'webllm',
      ls_model_name: 'test-model',
      ls_temperature: 0.5,
    })
  })

  it('preserves thinking through a real LangChain tool loop', async () => {
    const { model, create } = qwenModel()
    create.mockResolvedValueOnce(
      chunks(chunk({ content: '<think>Need the lookup.</think>' + toolOutput }, 'stop')),
    )
    create.mockResolvedValueOnce(
      chunks(chunk({ content: '<think>Use the result.</think>The answer is 42.' }, 'stop')),
    )
    const agent = createAgent({ model, tools: [lookup] })
    const result = await agent.invoke({ messages: [new HumanMessage('Look up A7.')] })
    expect(result.messages.map((message) => message.type)).toEqual(['human', 'ai', 'tool', 'ai'])
    const call = result.messages[1] as AIMessage
    expect(call.contentBlocks).toContainEqual(
      expect.objectContaining({ type: 'reasoning', reasoning: 'Need the lookup.' }),
    )
    expect(call.tool_calls?.[0]).toMatchObject({ name: 'lookup', args: { code: 'A7' } })
    expect(result.messages.at(-1)?.text).toBe('The answer is 42.')
    expect(create.mock.calls[1]?.[0]).toMatchObject({
      messages: expect.arrayContaining([
        expect.objectContaining({
          role: 'assistant',
          content: expect.stringContaining('<think>\nNeed the lookup.\n</think>'),
        }),
        { role: 'user', content: '<tool_response>\nResult A7: 42\n</tool_response>' },
      ]),
    })
  })

  it('uses the complete default Deep Agents middleware and built-in tools', async () => {
    const { model, create } = qwenModel()
    create.mockResolvedValueOnce(
      chunks(
        chunk(
          {
            content:
              '<think>Read the file.</think><tool_call>\n<function=read_file>\n<parameter=file_path>\n/answer.txt\n</parameter>\n</function>\n</tool_call>',
          },
          'stop',
        ),
      ),
    )
    create.mockResolvedValueOnce(
      chunks(chunk({ content: '<think>The file says 42.</think>The answer is 42.' }, 'stop')),
    )
    const agent = createDeepAgent({ model })
    const result = await agent.invoke({
      messages: [{ role: 'user', content: 'Read /answer.txt and tell me the answer.' }],
      files: {
        '/answer.txt': { content: ['42'], created_at: '2026-10-19', modified_at: '2026-10-19' },
      },
    })
    expect(result.messages.at(-1)?.text).toBe('The answer is 42.')
    const call = result.messages.find(
      (message) => AIMessage.isInstance(message) && message.tool_calls?.length,
    ) as AIMessage
    expect(call.tool_calls?.[0]).toMatchObject({
      name: 'read_file',
      args: { file_path: '/answer.txt' },
    })
    const first = create.mock.calls[0]?.[0]
    expect(first?.messages[0]?.content).toContain('write_file')
    expect(first?.messages[0]?.content).toContain('task')
    expect(
      create.mock.calls[1]?.[0].messages.some(
        (message) =>
          message.role === 'user' &&
          typeof message.content === 'string' &&
          message.content.includes('42'),
      ),
    ).toBe(true)
  })

  it('accepts Zod tools and structured output through the standard API', async () => {
    const { model, create } = qwenModel()
    create.mockResolvedValueOnce(
      chunks(
        chunk(
          {
            content:
              '<tool_call>\n<function=extract>\n<parameter=answer>\n42\n</parameter>\n</function>\n</tool_call>',
          },
          'stop',
        ),
      ),
    )
    expect(
      await model
        .withStructuredOutput(z.object({ answer: z.number() }))
        .invoke('Return the answer.'),
    ).toEqual({ answer: 42 })
    expect(() => model.bindTools([lookup, lookup])).toThrow('unique')
  })

  it('splits thinking without putting reasoning into answer text', async () => {
    const { model } = qwenModel(chunk({ content: '<think>A thought.</think>Answer' }, 'stop'))
    let result: AIMessageChunk | undefined
    for await (const item of await model.stream('Hi')) result = result ? result.concat(item) : item
    expect(result?.text).toBe('Answer')
    expect(result?.contentBlocks).toContainEqual(
      expect.objectContaining({ type: 'reasoning', reasoning: 'A thought.' }),
    )
  })

  it('does not replay thoughts from past user turns or change stored history', async () => {
    const { model, create } = qwenModel(chunk({ content: 'Okay' }, 'stop'))
    const history = [
      new HumanMessage('One'),
      new AIMessage({
        content: [
          { type: 'reasoning', reasoning: 'Old thought' },
          { type: 'text', text: 'One answer' },
        ],
      }),
      new HumanMessage('Two'),
    ]
    await model.invoke(history)
    expect(JSON.stringify(create.mock.calls[0]?.[0])).not.toContain('Old thought')
    expect(history[1]?.content).toContainEqual({ type: 'reasoning', reasoning: 'Old thought' })
  })

  it('preserves system messages, tool IDs, image and audio content', async () => {
    const { model, create } = setup(chunk({ content: 'Okay' }, 'stop'))
    await model.invoke([
      new SystemMessage('One'),
      new SystemMessage('Two'),
      new HumanMessage({
        content: [
          { type: 'text', text: 'First' },
          { type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } },
          { type: 'text', text: 'Last' },
          { type: 'input_audio', input_audio: { format: 'wav', data: 'AQ==' } },
        ],
      }),
      new AIMessage({
        content: '',
        tool_calls: [{ name: 'lookup', id: 'call-7', args: { code: 'A7' } }],
      }),
      new ToolMessage({ content: '42', tool_call_id: 'call-7' }),
    ])
    expect(create.mock.calls[0]?.[0]).toMatchObject({
      messages: [
        { role: 'system', content: 'One\n\nTwo' },
        {
          role: 'user',
          content: [
            { type: 'text', text: 'First' },
            { type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } },
            { type: 'text', text: 'Last' },
            { type: 'input_audio', input_audio: { format: 'wav', data: 'AQ==' } },
          ],
        },
        {
          role: 'assistant',
          tool_calls: [{ id: 'call-7', function: { name: 'lookup', arguments: '{"code":"A7"}' } }],
        },
        { role: 'tool', content: '42', tool_call_id: 'call-7' },
      ],
    })
  })

  it.each(['video', 'file', 'unknown'])(
    'rejects unsupported %s content instead of losing it',
    async (type) => {
      const { model, create } = setup()
      await expect(model.invoke([new HumanMessage({ content: [{ type }] })])).rejects.toThrow(
        'cannot use',
      )
      expect(create).not.toHaveBeenCalled()
    },
  )

  it('handles empty final messages and rejects a missing stop marker', async () => {
    const { model, create } = setup(chunk({}, 'stop'))
    expect((await model.invoke('Hi')).text).toBe('')
    create.mockResolvedValueOnce(chunks(chunk({ content: 'Unfinished' })))
    await expect(model.invoke('Hi')).rejects.toThrow('without a stop')
  })

  it('never runs a truncated native tool call', async () => {
    const { model } = setup(
      chunk(
        {
          tool_calls: [
            {
              index: 0,
              id: 'call',
              type: 'function',
              function: { name: 'lookup', arguments: '{"code":"A7"}' },
            },
          ],
        },
        'length',
      ),
    )
    const result = await model.bindTools([lookup]).invoke('Look up A7.')
    expect(result.tool_calls).toEqual([])
    expect(result.response_metadata.finish_reason).toBe('length')
  })

  it('keeps tool_choice none and rejects unsupported required choices', async () => {
    const { model, create, engine } = setup(chunk({ content: 'Answer' }, 'stop'))
    await model.bindTools([lookup], { tool_choice: 'none' }).invoke('Hi')
    expect(create.mock.calls[0]?.[0]).not.toHaveProperty('tools')
    expect(create.mock.calls[0]?.[0].tool_choice).toBe('none')
    await expect(
      model.bindTools([lookup], { tool_choice: 'required' }).invoke('Hi'),
    ).rejects.toThrow('required')
    const gemma = new ChatWebLLM({ engine, modelId: 'gemma-4-E2B-it-q4f16_1-MLC' })
    await expect(
      gemma.bindTools([lookup], { tool_choice: 'required' }).invoke('Hi'),
    ).rejects.toThrow('required')
  })
})
