import { Agent, AfterToolsEvent, FunctionTool } from '@strands-agents/sdk'
import { describe, expect, it } from 'vitest'
import { chunk, setup } from '../../../packages/strands-webllm/test/helpers.js'
import { runAgent, type ExampleInput } from '../src/output.js'

describe('example output', () => {
  it('shows a final hook message when there were no text deltas', async () => {
    const { engine, model } = setup(
      chunk(
        { tool_calls: [{ index: 0, id: 'clock-1', function: { name: 'clock', arguments: '{}' } }] },
        'tool_calls',
      ),
    )
    const agent = new Agent({
      model,
      printer: false,
      tools: [
        new FunctionTool({
          name: 'clock',
          description: 'Get time',
          callback: () => '12:00',
        }),
      ],
    })
    agent.addHook(AfterToolsEvent, (event) => {
      event.endTurn = 'Finished'
    })
    let output = ''
    const details: unknown[] = []
    const input: ExampleInput = {
      engine,
      modelId: 'test-model',
      prompt: 'Time?',
      file: undefined,
      signal: new AbortController().signal,
      write: (text) => {
        output += text
      },
      info: (value) => details.push(value),
    }
    expect(await runAgent(agent, input.prompt, input)).toBe('Finished')
    expect(output).toBe('Finished')
    expect(details).toEqual(
      expect.arrayContaining([expect.objectContaining({ type: 'toolResultEvent' })]),
    )
  })

  it('does not repeat streamed text when the final Agent result arrives', async () => {
    const { engine, model } = setup(chunk({ content: 'Hello' }, 'stop'))
    let output = ''
    const input: ExampleInput = {
      engine,
      modelId: 'test-model',
      prompt: 'Hi',
      file: undefined,
      signal: new AbortController().signal,
      write: (text) => {
        output += text
      },
      info: () => {},
    }
    expect(await runAgent(new Agent({ model, printer: false }), 'Hi', input)).toBe('Hello')
    expect(output).toBe('Hello')
  })
})
