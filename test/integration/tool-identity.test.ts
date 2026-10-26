import { expect, it, vi } from 'vitest'
import { Agent, FunctionTool, Message, TextBlock, ToolUseBlock } from '@strands-agents/sdk'
import { createAgent } from 'langchain'
import { tool } from '@langchain/core/tools'
import { z } from 'zod'
import { WebLLMModel } from '../../packages/strands-web-llm/src/index.js'
import { ChatWebLLM } from '../../packages/langchain-web-llm/src/index.js'
import { chunk, chunks, collect, setupEngine } from '../../fixtures/web-llm.js'

for (const client of ['strands', 'langchain']) {
  it(`${client} lets the framework handle an unknown native tool call`, async () => {
    const { engine, create } = setupEngine(chunk({ content: 'Done' }, 'stop'))
    create.mockResolvedValueOnce(
      chunks(
        chunk(
          {
            tool_calls: [
              {
                index: 0,
                id: 'call-1',
                function: { name: 'unregistered', arguments: '{}' },
              },
            ],
          },
          'tool_calls',
        ),
      ),
    )
    create.mockResolvedValueOnce(
      chunks(
        chunk(
          {
            tool_calls: [
              { index: 0, id: 'call-2', function: { name: 'weather', arguments: '{}' } },
            ],
          },
          'tool_calls',
        ),
      ),
    )
    const modelId = 'test-model'
    const weather = vi.fn(() => 'Sunny')

    if (client === 'strands') {
      const model = new WebLLMModel({ engine, modelId })
      const agent = new Agent({
        model,
        printer: false,
        tools: [
          new FunctionTool({ name: 'weather', description: 'Get the weather.', callback: weather }),
        ],
      })
      const result = await agent.invoke('Run a tool.')
      expect(result.lastMessage.content).toEqual([expect.objectContaining({ text: 'Done' })])
      expect(weather).toHaveBeenCalledOnce()
      expect(create).toHaveBeenCalledTimes(3)
      expect(create.mock.calls[1]?.[0].messages).toContainEqual(
        expect.objectContaining({ role: 'tool', tool_call_id: 'call-1' }),
      )
    } else {
      const model = new ChatWebLLM({ engine, modelId })
      const agent = createAgent({
        model,
        tools: [
          tool(weather, { name: 'weather', description: 'Get the weather.', schema: z.object({}) }),
        ],
      })
      const result = await agent.invoke({ messages: [{ role: 'user', content: 'Run a tool.' }] })
      expect(result.messages.at(-1)).toMatchObject({
        type: 'tool',
        tool_call_id: 'call-1',
        status: 'error',
        content: expect.stringContaining('unregistered'),
      })
      expect(weather).not.toHaveBeenCalled()
    }
  })

  it.each(['together', 'separate'])(
    `${client} rejects duplicate tool IDs in %s chunks`,
    async (mode) => {
      const { engine, create } = setupEngine(chunk({ content: 'Recovered' }, 'stop'))
      const calls = [0, 1].map((index) => ({
        index,
        id: 'shared',
        function: { name: 'weather', arguments: '{"city":"Seoul"}' },
      }))
      create.mockResolvedValueOnce(
        chunks(
          ...(mode === 'together'
            ? [chunk({ tool_calls: calls }, 'tool_calls')]
            : [
                chunk({ tool_calls: [calls[0]!] }),
                chunk({ tool_calls: [calls[1]!] }, 'tool_calls'),
              ]),
        ),
      )
      const options = { engine, modelId: 'test-model' }
      if (client === 'strands') {
        const model = new WebLLMModel(options)
        await expect(
          collect(
            model.streamAggregated([], {
              toolSpecs: [{ name: 'weather', description: 'Get the weather.' }],
            }),
          ),
        ).rejects.toThrow('duplicate tool call ID')
        expect(await collect(model.stream([]))).toContainEqual({
          type: 'modelContentBlockDeltaEvent',
          delta: { type: 'textDelta', text: 'Recovered' },
        })
      } else {
        const model = new ChatWebLLM(options)
        await expect(
          model
            .bindTools([{ type: 'function', function: { name: 'weather' } }])
            .invoke('Weather in Seoul?'),
        ).rejects.toThrow('duplicate tool call ID')
        expect((await model.invoke('Try again.')).text).toBe('Recovered')
      }
    },
  )

  it.each([
    { name: 'changed', first: 'call-1', last: 'call-2', invalid: true },
    { name: 'repeated', first: 'call-1', last: 'call-1', invalid: false },
    { name: 'late', first: undefined, last: 'call-1', invalid: false },
    { name: 'omitted', first: 'call-1', last: undefined, invalid: false },
  ])(`${client} handles a $name native tool ID`, async ({ first, last, invalid }) => {
    const { engine, create } = setupEngine(chunk({ content: 'Recovered' }, 'stop'))
    create.mockResolvedValueOnce(
      chunks(
        chunk({
          tool_calls: [
            {
              index: 0,
              ...(first && { id: first }),
              function: { name: 'wea', arguments: '{"city":' },
            },
          ],
        }),
        chunk(
          {
            tool_calls: [
              {
                index: 0,
                ...(last && { id: last }),
                function: { name: 'ther', arguments: '"Seoul"}' },
              },
            ],
          },
          'tool_calls',
        ),
      ),
    )
    const strands = new WebLLMModel({ engine, modelId: 'test-model' })
    const langchain = new ChatWebLLM({ engine, modelId: 'test-model' })
    const result = async () => {
      if (client === 'strands') {
        const blocks = await collect(
          strands.streamAggregated(
            [new Message({ role: 'user', content: [new TextBlock('Weather in Seoul?')] })],
            { toolSpecs: [{ name: 'weather', description: 'Get the weather.' }] },
          ),
        )
        return blocks
          .filter((block) => block instanceof ToolUseBlock)
          .map((block) => ({ id: block.toolUseId, name: block.name, input: block.input }))
      }
      const message = await langchain
        .bindTools([
          { type: 'function', function: { name: 'weather', description: 'Get the weather.' } },
        ])
        .invoke('Weather in Seoul?')
      return message.tool_calls?.map((call) => ({ id: call.id, name: call.name, input: call.args }))
    }
    if (invalid) await expect(result()).rejects.toThrow('changed a tool call ID')
    else
      expect(await result()).toEqual([
        { id: last ?? first, name: 'weather', input: { city: 'Seoul' } },
      ])
    if (client === 'strands') {
      const events = await collect(strands.stream([]))
      expect(events).toContainEqual({
        type: 'modelContentBlockDeltaEvent',
        delta: { type: 'textDelta', text: 'Recovered' },
      })
    } else expect((await langchain.invoke('Try again.')).text).toBe('Recovered')
  })
}
