import { expect, it } from 'vitest'
import { HumanMessage, SystemMessage } from '@langchain/core/messages'
import { Message, TextBlock } from '@strands-agents/sdk'
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript'
import {
  ModelType,
  postInitAndCheckFieldsChatCompletion,
  type ChatCompletionRequestStreaming,
} from '@mlc-ai/web-llm'
import { WebLLMModel } from '../../packages/strands-web-llm/src/index.js'
import { ChatWebLLM } from '../../packages/langchain-web-llm/src/index.js'
import { createWebLLMProvider } from '../../packages/pi-ai-web-llm/src/index.js'
import { chunk, collect, setupEngine } from '../../fixtures/web-llm.js'

const hermes = 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC'
const lookup = { name: 'lookup', description: 'Look up a code.' }

it.each(['strands', 'langchain', 'pi'] as const)(
  '%s disables native Hermes tools before WebLLM validates the request',
  async (client) => {
    const { engine, create } = setupEngine(chunk({ content: 'Done' }, 'stop'))
    const prompt = 'Answer briefly.'
    const tool = { ...lookup, parameters: { type: 'object', properties: {} } }
    if (client === 'strands') {
      const model = new WebLLMModel({ engine, modelId: hermes, request: { tool_choice: 'none' } })
      await collect(
        model.stream([new Message({ role: 'user', content: [new TextBlock('Hi')] })], {
          systemPrompt: prompt,
          toolSpecs: [lookup],
        }),
      )
    } else if (client === 'langchain') {
      const model = new ChatWebLLM({ engine, modelId: hermes })
      await model
        .bindTools([{ type: 'function', function: tool }], { tool_choice: 'none' })
        .invoke([new SystemMessage(prompt), new HumanMessage('Hi')])
    } else {
      const provider = createWebLLMProvider({ engine, modelId: hermes, contextWindow: 8192 })
      const context = normalizeContext({
        systemPrompt: prompt,
        tools: [tool],
        messages: [{ role: 'user', content: 'Hi', timestamp: 0 }],
      })
      const result = await provider
        .stream(provider.getModels()[0]!, context, {
          toolChoice: 'none',
        })
        .result()
      expect(result.stopReason).toBe('stop')
    }
    const value = create.mock.calls[0]![0] as ChatCompletionRequestStreaming
    expect(value.tool_choice).toBe('none')
    expect(value.tools).toBeUndefined()
    expect(value.messages[0]).toEqual({ role: 'system', content: prompt })
    const checked = structuredClone(value)
    expect(() => postInitAndCheckFieldsChatCompletion(checked, hermes, ModelType.LLM)).not.toThrow()
    expect(checked.messages[0]).toEqual({ role: 'system', content: prompt })
    expect(checked.response_format).toBeUndefined()
  },
)

async function request(client: string, modelId: string, prompt: string | undefined, tools = false) {
  const { engine, create } = setupEngine(chunk({ content: 'Done' }, 'stop'))
  if (client === 'strands') {
    const model = new WebLLMModel({ engine, modelId })
    await collect(
      model.stream([new Message({ role: 'user', content: [new TextBlock('Hi')] })], {
        ...(prompt !== undefined && { systemPrompt: prompt }),
        ...(tools && { toolSpecs: [lookup] }),
      }),
    )
  } else {
    const model = new ChatWebLLM({ engine, modelId })
    const llm = tools
      ? model.bindTools([
          {
            type: 'function',
            function: { ...lookup, parameters: { type: 'object', properties: {} } },
          },
        ])
      : model
    await llm.invoke([
      ...(prompt !== undefined ? [new SystemMessage(prompt)] : []),
      new HumanMessage('Hi'),
    ])
  }
  return create.mock.calls[0]![0] as ChatCompletionRequestStreaming
}

for (const client of ['strands', 'langchain']) {
  for (const modelId of [
    'test-model',
    hermes,
    'Qwen2.5-3B-Instruct-q4f16_1-MLC',
    'Qwen3-4B-q4f16_1-MLC',
    'Qwen3.5-4B-q4f16_1-MLC',
  ]) {
    it.each([undefined, '', ' ', 'Use these rules.'])(
      `${client} preserves the system prompt %j without native tools (${modelId})`,
      async (prompt) => {
        const value = await request(client, modelId, prompt)
        const system =
          prompt ??
          (modelId.startsWith('Qwen2.5-')
            ? 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
            : undefined)
        expect(value.messages).toEqual([
          ...(system !== undefined ? [{ role: 'system', content: system }] : []),
          { role: 'user', content: 'Hi' },
        ])
      },
    )
  }

  it(`${client} clears the Qwen2.5 default with an empty system prompt`, async () => {
    const value = await request(client, 'Qwen2.5-3B-Instruct-q4f16_1-MLC', '')
    expect(value.messages).toEqual([
      { role: 'system', content: '' },
      { role: 'user', content: 'Hi' },
    ])
  })

  it.each(['Qwen3-4B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC'])(
    `${client} keeps the %s tool prompt without adding a default system message`,
    async (modelId) => {
      const value = await request(client, modelId, undefined, true)
      expect(value.messages[0]?.content).not.toContain('You are a helpful assistant.')
    },
  )

  it.each([undefined, '', ' ', 'Use these rules.'])(
    `${client} preserves the Qwen3.5 system prompt with tools: %j`,
    async (prompt) => {
      const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
      const base = await request(client, modelId, undefined, true)
      const value = await request(client, modelId, prompt, true)
      const declarations = base.messages[0]!.content as string

      expect(declarations).toContain('<tools>')
      expect(value.messages[0]).toEqual({
        role: 'system',
        content: prompt ? `${declarations}\n\n${prompt}` : declarations,
      })
    },
  )

  it(`${client} keeps its Qwen3 empty system prompt behavior with tools`, async () => {
    const modelId = 'Qwen3-4B-q4f16_1-MLC'
    const base = await request(client, modelId, undefined, true)
    const value = await request(client, modelId, '', true)
    const declarations = base.messages[0]!.content as string

    expect(value.messages[0]).toEqual({
      role: 'system',
      content: client === 'strands' ? `\n\n${declarations}` : declarations,
    })
  })

  it.each([undefined, ''])(
    `${client} allows Hermes to supply its native tool prompt: %j`,
    async (prompt) => {
      const value = await request(client, hermes, prompt, true)
      expect(value.messages).toEqual([{ role: 'user', content: 'Hi' }])
      expect(() =>
        postInitAndCheckFieldsChatCompletion(structuredClone(value), hermes, ModelType.LLM),
      ).not.toThrow()
      expect(value.messages).toEqual([{ role: 'user', content: 'Hi' }])
    },
  )

  it.each([' ', 'Use these rules.'])(
    `${client} keeps a custom Hermes tool prompt for native validation: %j`,
    async (prompt) => {
      const value = await request(client, hermes, prompt, true)
      expect(value.messages[0]).toEqual({ role: 'system', content: prompt })
      expect(() =>
        postInitAndCheckFieldsChatCompletion(structuredClone(value), hermes, ModelType.LLM),
      ).toThrow('cannot specify customized system prompt')
    },
  )
}
