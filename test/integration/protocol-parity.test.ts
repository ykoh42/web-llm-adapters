import { describe, expect, it, vi } from 'vitest'
import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages'
import {
  Message,
  TextBlock,
  ReasoningBlock,
  ToolUseBlock,
  ToolResultBlock,
} from '@strands-agents/sdk'
import { WebLLMModel } from '../../packages/strands-web-llm/src/index.js'
import { ChatWebLLM } from '../../packages/langchain-web-llm/src/index.js'
import { chunk, collect, setup } from '../../packages/langchain-web-llm/fixtures/engine.js'

const ids = [
  'Qwen3.5-9B-q4f16_1-MLC',
  'Qwen3.5-4B-q4f16_1-MLC',
  'Qwen3.5-2B-q4f16_1-MLC',
  'Qwen3.5-0.8B-q4f16_1-MLC',
  'Qwen3-8B-q4f16_1-MLC',
  'Qwen3-4B-q4f16_1-MLC',
  'Qwen3-0.6B-q4f16_1-MLC',
  'Qwen2.5-7B-Instruct-q4f16_1-MLC',
  'Qwen2.5-3B-Instruct-q4f16_1-MLC',
  'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
  'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
  'gemma-4-E2B-it-q4f16_1-MLC',
]
const schema = {
  type: 'object' as const,
  properties: { code: { type: 'string' as const } },
  required: ['code'],
}
const declaration = { name: 'lookup', description: 'Look up a code.', inputSchema: schema }
const nativeTool = {
  type: 'function',
  function: { name: declaration.name, description: declaration.description, parameters: schema },
}

describe('shared model protocols', () => {
  it.each(ids)(
    '%s sends the same canonical tool history through Strands and LangChain',
    async (modelId) => {
      const { engine, create } = setup(chunk({ content: '42' }, 'stop'))
      const raw = vi.spyOn(engine.completions, 'create').mockImplementation(async () =>
        (async function* () {
          yield {
            id: 'raw',
            object: 'text_completion' as const,
            created: 0,
            model: modelId,
            choices: [
              {
                index: 0,
                text: '<|channel>thought\nDone<channel|>42',
                finish_reason: 'stop' as const,
                logprobs: null,
              },
            ],
          }
        })(),
      )
      const request = {
        max_tokens: 4096,
        extra_body: { enable_thinking: !modelId.startsWith('Qwen2.5-') },
      }
      const strands = new WebLLMModel({ engine, modelId, request })
      const lc = new ChatWebLLM({ engine, modelId, request })
      const strandsHistory = [
        new Message({ role: 'user', content: [new TextBlock('Find A7')] }),
        new Message({
          role: 'assistant',
          content: [
            new ReasoningBlock({ text: 'Read the code' }),
            new ToolUseBlock({ toolUseId: 'call', name: 'lookup', input: { code: 'A7' } }),
          ],
        }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'call',
              content: [new TextBlock('42')],
              status: 'success',
            }),
          ],
        }),
      ]
      const lcHistory = [
        new SystemMessage('Caller system'),
        new HumanMessage('Find A7'),
        new AIMessage({
          content: [{ type: 'reasoning', reasoning: 'Read the code' }],
          tool_calls: [{ id: 'call', name: 'lookup', args: { code: 'A7' } }],
        }),
        new ToolMessage({ content: '42', tool_call_id: 'call' }),
      ]
      await collect(
        strands.stream(strandsHistory, { systemPrompt: 'Caller system', toolSpecs: [declaration] }),
      )
      await lc.bindTools([nativeTool]).invoke(lcHistory)
      if (modelId.startsWith('gemma-'))
        expect(raw.mock.calls[1]?.[0]).toEqual(raw.mock.calls[0]?.[0])
      else expect(create.mock.calls[1]?.[0]).toEqual(create.mock.calls[0]?.[0])
    },
  )
})
