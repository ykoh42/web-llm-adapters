import { Agent, FunctionTool, TextBlock } from '@strands-agents/sdk'
import { expect, it, vi } from 'vitest'
import { chunk, chunks, collect, setup } from '../../fixtures/engine.js'

it.each([
  { field: 'name', delta: { function: { name: '_other' } }, error: 'changed a tool name' },
  { field: 'ID', delta: { id: 'call-2' }, error: 'changed a tool call ID' },
])('rejects a changed native tool $field before Agent execution', async ({ delta, error }) => {
  const { model, create } = setup()
  create.mockResolvedValueOnce(
    chunks(
      chunk({
        tool_calls: [
          { index: 0, id: 'call-1', function: { name: 'weather', arguments: '{"city":"Seoul"}' } },
        ],
      }),
      chunk({ tool_calls: [{ index: 0, ...delta }] }, 'tool_calls'),
    ),
  )
  create.mockResolvedValueOnce(chunks(chunk({ content: 'Sunny' }, 'stop')))
  const callback = vi.fn(() => 'Sunny')
  const agent = new Agent({
    model,
    printer: false,
    tools: [new FunctionTool({ name: 'weather', description: 'Get the weather.', callback })],
  })
  await expect(agent.invoke('Weather in Seoul?')).rejects.toThrow(error)
  expect(callback).not.toHaveBeenCalled()
  expect(create).toHaveBeenCalledTimes(1)
  const recovered = await collect(model.streamAggregated([]))
  expect(recovered.filter((block) => block instanceof TextBlock)).toEqual([new TextBlock('Sunny')])
})
