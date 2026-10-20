import { Agent, AfterToolsEvent, FunctionTool, TextBlock } from '@strands-agents/sdk'
import { describe, expect, it, vi } from 'vitest'
import { chunk, setup } from '../../../../../packages/strands-web-llm/fixtures/engine.js'
import { runAgent, type ExampleInput } from '../../src/examples/output.js'

describe('example output', () => {
  it('keeps reasoning in Details and only shows text in the answer', async () => {
    const { engine, model } = setup()
    vi.spyOn(model, 'stream').mockImplementation(async function* () {
      yield { type: 'modelMessageStartEvent', role: 'assistant' }
      yield { type: 'modelContentBlockStartEvent' }
      yield {
        type: 'modelContentBlockDeltaEvent',
        delta: { type: 'reasoningContentDelta', text: 'Check first.' },
      }
      yield { type: 'modelContentBlockStopEvent' }
      yield { type: 'modelContentBlockStartEvent' }
      yield { type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text: '42' } }
      yield { type: 'modelContentBlockStopEvent' }
      yield { type: 'modelMessageStopEvent', stopReason: 'endTurn' }
    })
    let output = ''
    const details: unknown[] = []
    const input: ExampleInput = {
      engine,
      modelId: 'test-model',
      prompt: 'Answer.',
      file: undefined,
      signal: new AbortController().signal,
      write: (text) => {
        output += text
      },
      info: (value) => details.push(value),
    }
    expect(await runAgent(new Agent({ model, printer: false }), input.prompt, input)).toBe('42')
    expect(output).toBe('42')
    expect(details).toContainEqual({ reasoning: ['Check first.'] })
  })
  it.each(['Finished', ''])('preserves the explicit hook answer %j', async (answer) => {
    const { engine, model, create } = setup(
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
      event.endTurn = [new TextBlock(answer)]
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
    expect(await runAgent(agent, input.prompt, input)).toBe(answer)
    expect(output).toBe(answer)
    expect(create).toHaveBeenCalledTimes(1)
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
