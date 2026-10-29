import { describe, expect, it, vi } from 'vitest'
import { Type, type AssistantMessage } from '@earendil-works/pi-ai'
import { createModels } from '@earendil-works/pi-ai/models'
import { AIMessage, HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages'
import {
  Message,
  CachePointBlock,
  TextBlock,
  ReasoningBlock,
  ToolUseBlock,
  ToolResultBlock,
} from '@strands-agents/sdk'
import { WebLLMModel } from '../../packages/strands-web-llm/src/index.js'
import { ChatWebLLM } from '../../packages/langchain-web-llm/src/index.js'
import { createWebLLMProvider } from '../../packages/pi-ai-web-llm/src/index.js'
import { chunk, collect, setupEngine } from '../../fixtures/web-llm.js'
import { hasToolResult } from '../support/native-output.js'

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

function record(modelId: string) {
  const { engine, create } = setupEngine(chunk({ content: '42' }, 'stop'))
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
  const provider = createWebLLMProvider({ engine, modelId, contextWindow: 32768, request })
  const models = createModels()
  models.setProvider(provider)
  const model = provider.getModels()[0]!
  return { engine, create, raw, request, models, model }
}

const piTool = {
  name: declaration.name,
  description: declaration.description,
  parameters: Type.Unsafe<{ code: string }>(schema),
}

function assistant(modelId: string, content: AssistantMessage['content']): AssistantMessage {
  return {
    role: 'assistant',
    api: 'webllm',
    provider: 'webllm',
    model: modelId,
    content,
    stopReason: 'stop',
    timestamp: 0,
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
  }
}

describe('shared model protocols', () => {
  for (const content of [[], [new CachePointBlock({ cacheType: 'default' })]]) {
    it.each(ids)(
      `%s preserves an empty user turn with ${content.length} cache hints`,
      async (modelId) => {
        const { engine, create, raw, request, models, model } = record(modelId)
        const strands = new WebLLMModel({ engine, modelId, request })
        const lc = new ChatWebLLM({ engine, modelId, request })
        const thought = new ReasoningBlock({ text: 'Earlier thought.' })
        const history = [
          new Message({ role: 'user', content: [new TextBlock('First question.')] }),
          new Message({ role: 'assistant', content: [thought, new TextBlock('Earlier answer.')] }),
          new Message({ role: 'user', content }),
        ]
        await collect(
          strands.stream(history, { systemPrompt: 'Caller system', toolSpecs: [declaration] }),
        )
        await lc.bindTools([nativeTool]).invoke([
          new SystemMessage('Caller system'),
          new HumanMessage('First question.'),
          new AIMessage({
            content: [
              { type: 'reasoning', reasoning: 'Earlier thought.' },
              { type: 'text', text: 'Earlier answer.' },
            ],
          }),
          new HumanMessage(''),
        ])
        const result = await models.complete(model, {
          systemPrompt: 'Caller system',
          tools: [piTool],
          messages: [
            { role: 'user', content: 'First question.', timestamp: 0 },
            assistant(modelId, [
              { type: 'thinking', thinking: 'Earlier thought.' },
              { type: 'text', text: 'Earlier answer.' },
            ]),
            { role: 'user', content: '', timestamp: 0 },
          ],
        })
        expect(result.stopReason).toBe('stop')
        if (modelId.startsWith('gemma-')) {
          expect(raw.mock.calls[1]?.[0]).toEqual(raw.mock.calls[0]?.[0])
          expect(raw.mock.calls[2]?.[0]).toEqual(raw.mock.calls[0]?.[0])
        } else {
          expect(create.mock.calls[1]?.[0]).toEqual(create.mock.calls[0]?.[0])
          expect(create.mock.calls[2]?.[0]).toEqual(create.mock.calls[0]?.[0])
        }
        expect(history[1]!.content[0]).toBe(thought)
        expect(history[2]!.content).toEqual(content)
      },
    )
  }

  it.each(ids)('%s sends the same canonical tool history through all adapters', async (modelId) => {
    const { engine, create, raw, request, models, model } = record(modelId)
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
    const result = await models.complete(model, {
      systemPrompt: 'Caller system',
      tools: [piTool],
      messages: [
        { role: 'user', content: 'Find A7', timestamp: 0 },
        {
          ...assistant(modelId, [
            { type: 'thinking', thinking: 'Read the code' },
            { type: 'toolCall', id: 'call', name: 'lookup', arguments: { code: 'A7' } },
          ]),
          stopReason: 'toolUse',
        },
        {
          role: 'toolResult',
          toolCallId: 'call',
          toolName: 'lookup',
          content: [{ type: 'text', text: '42' }],
          isError: false,
          timestamp: 0,
        },
      ],
    })
    expect(result.stopReason).toBe('stop')
    if (modelId.startsWith('gemma-')) {
      expect(raw.mock.calls[1]?.[0]).toEqual(raw.mock.calls[0]?.[0])
      expect(raw.mock.calls[2]?.[0]).toEqual(raw.mock.calls[0]?.[0])
      for (const [request] of raw.mock.calls) expect(hasToolResult([request], '42')).toBe(true)
    } else {
      expect(create.mock.calls[1]?.[0]).toEqual(create.mock.calls[0]?.[0])
      expect(create.mock.calls[2]?.[0]).toEqual(create.mock.calls[0]?.[0])
      for (const [request] of create.mock.calls) expect(hasToolResult([request], '42')).toBe(true)
    }
  })
})
