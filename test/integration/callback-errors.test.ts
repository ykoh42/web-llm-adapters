import { expect, it, vi } from 'vitest'
import type { Completion } from '@mlc-ai/web-llm'
import { Agent, FunctionTool, ModelError } from '@strands-agents/sdk'
import { createAgent } from 'langchain'
import { tool } from '@langchain/core/tools'
import { z } from 'zod'
import { WebLLMModel } from '../../packages/strands-web-llm/src/index.js'
import { ChatWebLLM } from '../../packages/langchain-web-llm/src/index.js'
import { chunk, setupEngine } from '../../fixtures/web-llm.js'

const schema = z.object({ code: z.string() })
const jsonCall = '<tool_call>{"name":"lookup","arguments":{"code":"A7"}}</tool_call>'
const protocols = [
  { modelId: 'test-model', before: '', call: '' },
  { modelId: 'Qwen2.5-3B-Instruct-q4f16_1-MLC', before: 'Before', call: jsonCall },
  { modelId: 'Qwen3-4B-q4f16_1-MLC', before: '<think>Plan.</think>', call: jsonCall },
  {
    modelId: 'Qwen3.5-4B-q4f16_1-MLC',
    before: '<think>Plan.</think>',
    call: '<tool_call><function=lookup><parameter=code>\nA7\n</parameter></function></tool_call>',
  },
  {
    modelId: 'gemma-4-E2B-it-q4f16_1-MLC',
    before: '<|channel>thought\nPlan.<channel|>',
    call: '<|tool_call>call:lookup{code:<|"|>A7<|"|>}<tool_call|>',
  },
]

function raw(text: string, finish: 'stop' | null = null): Completion {
  return {
    id: 'test',
    model: 'test-model',
    object: 'text_completion',
    created: 0,
    choices: [{ index: 0, text, finish_reason: finish }],
  }
}

for (const framework of ['strands', 'langchain']) {
  for (const failAt of [1, 2]) {
    it.each(protocols)(
      `${framework} blocks $modelId tools when callback ${failAt} fails and recovers`,
      async ({ modelId, before, call }) => {
        const { engine, create, stop } = setupEngine(chunk({ content: 'Recovered' }, 'stop'))
        let closed = false
        async function* source<T>(values: T[]) {
          try {
            yield* values
          } finally {
            closed = true
          }
        }
        if (modelId.startsWith('gemma-')) {
          vi.spyOn(engine.completions, 'create')
            .mockImplementation(async () => source([raw('Recovered', 'stop')]))
            .mockResolvedValueOnce(source([raw(before), raw(call, 'stop')]))
        } else {
          const native = modelId === 'test-model'
          create.mockResolvedValueOnce(
            source(
              native
                ? [
                    chunk({
                      tool_calls: [
                        {
                          index: 0,
                          id: 'call-1',
                          function: { name: 'lookup', arguments: '{"code":"A7"}' },
                        },
                      ],
                    }),
                    chunk({}, 'tool_calls'),
                  ]
                : [chunk({ content: before }), chunk({ content: call }, 'stop')],
            ),
          )
        }
        const failure = new Error('Callback failed')
        const onChunk = vi.fn(() => {
          if (onChunk.mock.calls.length === failAt) throw failure
        })
        const callback = vi.fn(() => '42')
        const options = {
          engine,
          modelId,
          onChunk,
          request: { extra_body: { enable_thinking: true } },
        }
        let invoke: (input: string) => Promise<unknown>
        if (framework === 'strands') {
          const agent = new Agent({
            model: new WebLLMModel(options),
            tools: [new FunctionTool({ name: 'lookup', description: 'Look up a code.', callback })],
            printer: false,
          })
          invoke = async (input) => String(await agent.invoke(input))
        } else {
          const agent = createAgent({
            model: new ChatWebLLM(options),
            tools: [tool(callback, { name: 'lookup', description: 'Look up a code.', schema })],
          })
          invoke = async (input) =>
            (await agent.invoke({ messages: [{ role: 'user', content: input }] })).messages.at(-1)
              ?.text
        }
        const error: unknown = await invoke('Find A7.').catch((error: unknown) => error)
        if (framework === 'strands') {
          expect(error).toBeInstanceOf(ModelError)
          if (error instanceof ModelError) expect(error.cause).toBe(failure)
        } else expect(error).toBe(failure)
        expect(callback).not.toHaveBeenCalled()
        expect(onChunk).toHaveBeenCalledTimes(failAt)
        expect(closed).toBe(true)
        expect(stop).toHaveBeenCalledOnce()
        await expect(invoke('Try again.')).resolves.toBe('Recovered')
        expect(callback).not.toHaveBeenCalled()
      },
    )
  }
}
