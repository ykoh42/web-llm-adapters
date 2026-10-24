import { expect, it } from 'vitest'
import { Message, TextBlock, ToolUseBlock } from '@strands-agents/sdk'
import { WebLLMModel } from '../../packages/strands-web-llm/src/index.js'
import { ChatWebLLM } from '../../packages/langchain-web-llm/src/index.js'
import { chunk, chunks, collect, setupEngine } from '../../fixtures/web-llm.js'

for (const client of ['strands', 'langchain']) {
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
