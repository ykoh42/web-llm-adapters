import { expect, it } from 'vitest'
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages'
import { Message, TextBlock, ToolUseBlock, ToolResultBlock } from '@strands-agents/sdk'
import { WebLLMModel } from '../../packages/strands-web-llm/src/index.js'
import { ChatWebLLM } from '../../packages/langchain-web-llm/src/index.js'
import { chunk, collect, setup } from '../../packages/langchain-web-llm/fixtures/engine.js'

const modelId = 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC'
const original = {
  code: 'A7',
  nested: [{ left: 1, right: 2 }, null, false],
  ['__proto__']: { left: 1, right: 2 },
}
const reordered = {
  ['__proto__']: { right: 2, left: 1 },
  nested: [{ right: 2, left: 1 }, null, false],
  code: 'A7',
}

for (const client of ['strands', 'langchain']) {
  it.each([
    { input: reordered, same: true },
    { input: { ...reordered, nested: [false, null, { right: 2, left: 1 }] }, same: false },
    { input: { ...reordered, nested: [] }, same: false },
    { input: { ...reordered, nested: {} }, same: false },
    { input: { ...reordered, nested: [{ left: 1, other: 2 }, null, false] }, same: false },
    { input: { ...reordered, nested: [{ left: 1, right: 2 }, null, 0] }, same: false },
  ])(
    `${client} keeps native history only when JSON arguments match: %j`,
    async ({ input, same }) => {
      const raw = JSON.stringify([{ name: 'lookup', arguments: original, note: 'Keep this note.' }])
      const { engine, create } = setup(chunk({ content: 'Done' }, 'stop'))
      if (client === 'strands') {
        const model = new WebLLMModel({ engine, modelId })
        await collect(
          model.stream([
            new Message({ role: 'user', content: [new TextBlock('Look up A7.')] }),
            new Message({
              role: 'assistant',
              content: [
                new TextBlock(raw),
                new ToolUseBlock({ toolUseId: 'call', name: 'lookup', input }),
              ],
            }),
            new Message({
              role: 'user',
              content: [
                new ToolResultBlock({
                  toolUseId: 'call',
                  status: 'success',
                  content: [new TextBlock('Done.')],
                }),
              ],
            }),
          ]),
        )
      } else {
        const model = new ChatWebLLM({ engine, modelId })
        await model.invoke([
          new HumanMessage('Look up A7.'),
          new AIMessage({
            content: raw,
            tool_calls: [{ id: 'call', name: 'lookup', args: input }],
          }),
          new ToolMessage({ tool_call_id: 'call', content: 'Done.' }),
        ])
      }
      expect(create.mock.calls[0]?.[0].messages[1]?.content).toBe(
        same ? raw : raw + JSON.stringify([{ name: 'lookup', arguments: input }]),
      )
    },
  )

  it.each([false, true])(
    `${client} keeps extra native fields while suppressing duplicate calls: %s`,
    async (extra) => {
      const raw = JSON.stringify([
        { name: 'lookup', arguments: original, ...(extra && { note: 'Keep this note.' }) },
      ])
      const { engine } = setup(
        chunk({ content: raw }),
        chunk(
          {
            tool_calls: [
              { index: 0, function: { name: 'lookup', arguments: JSON.stringify(reordered) } },
            ],
          },
          'tool_calls',
        ),
      )
      let text = ''
      const calls = []
      if (client === 'strands') {
        const model = new WebLLMModel({ engine, modelId })
        const messages = [new Message({ role: 'user', content: [new TextBlock('Look up A7.')] })]
        for await (const block of model.streamAggregated(messages, {
          toolSpecs: [{ name: 'lookup', description: 'Look up a code.' }],
        })) {
          if (block instanceof TextBlock) text += block.text
          if (block instanceof ToolUseBlock) calls.push(block.input)
        }
      } else {
        const model = new ChatWebLLM({ engine, modelId })
        const result = await model
          .bindTools([{ type: 'function', function: { name: 'lookup' } }])
          .invoke('Look up A7.')
        text = result.text
        calls.push(...(result.tool_calls?.map((call) => call.args) ?? []))
      }
      expect(text).toBe(extra ? raw : '')
      expect(calls).toEqual([reordered])
    },
  )
}
