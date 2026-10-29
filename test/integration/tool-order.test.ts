import { expect, it } from 'vitest'
import { Agent as StrandsAgent, FunctionTool } from '@strands-agents/sdk'
import { Agent as PiAgent } from '@earendil-works/pi-agent-core'
import { Type } from '@earendil-works/pi-ai'
import { createModels } from '@earendil-works/pi-ai/models'
import { tool } from '@langchain/core/tools'
import { createAgent } from 'langchain'
import { z } from 'zod'
import { WebLLMModel } from '../../packages/strands-web-llm/src/index.js'
import { ChatWebLLM } from '../../packages/langchain-web-llm/src/index.js'
import { createWebLLMProvider } from '../../packages/pi-ai-web-llm/src/index.js'
import { chunk, chunks, pause, setupEngine } from '../../fixtures/web-llm.js'

const calls = ['slow', 'fast']
  .map((name) => `<tool_call>{"name":"${name}","arguments":{}}</tool_call>`)
  .join('')

for (const [protocol, modelId] of [
  ['Qwen', 'Qwen3-4B-q4f16_1-MLC'],
  ['native', 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC'],
] as const) {
  for (const framework of ['strands', 'langchain', 'pi'] as const) {
    it(`${framework} keeps parallel results in call order for the next ${protocol} request`, async () => {
      const { engine, create } = setupEngine(chunk({ content: 'Done' }, 'stop'))
      const response =
        protocol === 'Qwen'
          ? chunk({ content: calls }, 'stop')
          : chunk(
              {
                tool_calls: ['slow', 'fast'].map((name, index) => ({
                  index,
                  id: `call-${index}`,
                  function: { name, arguments: '{}' },
                })),
              },
              'tool_calls',
            )
      create.mockResolvedValueOnce(chunks(response))
      const gate = pause()
      const finished: string[] = []
      const slow = async () => {
        await gate.wait
        finished.push('slow')
        return 'slow result'
      }
      const fast = async () => {
        finished.push('fast')
        gate.resume()
        return 'fast result'
      }

      if (framework === 'strands') {
        const agent = new StrandsAgent({
          model: new WebLLMModel({ engine, modelId }),
          printer: false,
          tools: [
            new FunctionTool({ name: 'slow', description: 'Slow tool.', callback: slow }),
            new FunctionTool({ name: 'fast', description: 'Fast tool.', callback: fast }),
          ],
        })
        await agent.invoke('Use both tools.')
      } else if (framework === 'langchain') {
        const agent = createAgent({
          model: new ChatWebLLM({ engine, modelId }),
          tools: [
            tool(slow, { name: 'slow', description: 'Slow tool.', schema: z.object({}) }),
            tool(fast, { name: 'fast', description: 'Fast tool.', schema: z.object({}) }),
          ],
        })
        await agent.invoke({ messages: [{ role: 'user', content: 'Use both tools.' }] })
      } else {
        const provider = createWebLLMProvider({ engine, modelId, contextWindow: 8192 })
        const models = createModels()
        models.setProvider(provider)
        const agent = new PiAgent({
          initialState: {
            model: provider.getModels()[0]!,
            tools: [
              {
                name: 'slow',
                label: 'Slow',
                description: 'Slow tool.',
                parameters: Type.Object({}),
                execute: async () => ({
                  content: [{ type: 'text', text: await slow() }],
                  details: {},
                }),
              },
              {
                name: 'fast',
                label: 'Fast',
                description: 'Fast tool.',
                parameters: Type.Object({}),
                execute: async () => ({
                  content: [{ type: 'text', text: await fast() }],
                  details: {},
                }),
              },
            ],
          },
          streamFn: models.streamSimple.bind(models),
        })
        await agent.prompt('Use both tools.')
      }

      expect(finished).toEqual(['fast', 'slow'])
      const messages = create.mock.calls[1]?.[0].messages
      if (protocol === 'Qwen') {
        const result = messages?.filter((message) => message.role === 'user').at(-1)
        expect(result?.content).toBe(
          '<tool_response>\nslow result\n</tool_response>\n<tool_response>\nfast result\n</tool_response>',
        )
      } else {
        expect(messages?.filter((message) => message.role === 'tool')).toEqual([
          { role: 'tool', tool_call_id: 'call-0', content: 'slow result' },
          { role: 'tool', tool_call_id: 'call-1', content: 'fast result' },
        ])
      }
    })
  }
}
