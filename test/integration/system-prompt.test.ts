import { expect, it } from 'vitest'
import { HumanMessage, SystemMessage } from '@langchain/core/messages'
import { Message, TextBlock } from '@strands-agents/sdk'
import {
  ModelType,
  postInitAndCheckFieldsChatCompletion,
  type ChatCompletionRequestStreaming,
} from '@mlc-ai/web-llm'
import { WebLLMModel } from '../../packages/strands-web-llm/src/index.js'
import { ChatWebLLM } from '../../packages/langchain-web-llm/src/index.js'
import { chunk, collect, setup } from '../../packages/langchain-web-llm/fixtures/engine.js'

const hermes = 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC'
const lookup = { name: 'lookup', description: 'Look up a code.' }

async function request(client: string, modelId: string, prompt: string | undefined, tools = false) {
  const { engine, create } = setup(chunk({ content: 'Done' }, 'stop'))
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
  for (const modelId of ['test-model', hermes, 'Qwen3-4B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC']) {
    it.each([undefined, '', ' ', 'Use these rules.'])(
      `${client} preserves the system prompt %j without native tools (${modelId})`,
      async (prompt) => {
        const value = await request(client, modelId, prompt)
        expect(value.messages).toEqual([
          ...(prompt !== undefined ? [{ role: 'system', content: prompt }] : []),
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
