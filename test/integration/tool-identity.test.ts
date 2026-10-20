import { expect, it } from 'vitest'
import { Message, TextBlock, ToolUseBlock } from '@strands-agents/sdk'
import { WebLLMModel } from '../../packages/strands-web-llm/src/index.js'
import { ChatWebLLM } from '../../packages/langchain-web-llm/src/index.js'
import { chunk, chunks, collect, setup } from '../../packages/langchain-web-llm/fixtures/engine.js'

for (const client of ['strands', 'langchain']) {
  it.each([
    { name: 'changed', first: 'call-1', last: 'call-2', invalid: true },
    { name: 'repeated', first: 'call-1', last: 'call-1', invalid: false },
    { name: 'late', first: undefined, last: 'call-1', invalid: false },
    { name: 'omitted', first: 'call-1', last: undefined, invalid: false },
  ])(`${client} handles a $name native tool ID`, async ({ first, last, invalid }) => {
    const { engine, create } = setup(chunk({ content: 'Recovered' }, 'stop'))
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
