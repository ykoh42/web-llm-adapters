import { expect, it } from 'vitest'
import { Agent, MaxTokensError } from '@strands-agents/sdk'
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript'
import { ChatWebLLM } from '../../packages/langchain-web-llm/src/index.js'
import { createWebLLMProvider } from '../../packages/pi-ai-web-llm/src/index.js'
import { WebLLMModel } from '../../packages/strands-web-llm/src/index.js'
import { chunk, setupEngine } from '../../fixtures/web-llm.js'

const modelId = 'Qwen3.5-4B-q4f16_1-MLC'
const thinking = 'The answer is still being considered.'
const request = { extra_body: { enable_thinking: true } }

it('keeps unfinished thinking in a Strands agent limit error', async () => {
  const { engine, create } = setupEngine(chunk({ content: thinking }, 'length'))
  const model = new WebLLMModel({ engine, modelId, thinkingPrefilled: true, request })
  const result = await new Agent({ model, printer: false })
    .invoke('Hello')
    .catch((error: unknown) => error)

  expect(result).toBeInstanceOf(MaxTokensError)
  expect(result).toMatchObject({
    partialMessage: {
      content: [expect.objectContaining({ type: 'reasoningBlock', text: thinking })],
    },
  })
  expect(create).toHaveBeenCalledOnce()
})

it('keeps unfinished thinking and the limit reason in a LangChain result', async () => {
  const { engine, create } = setupEngine(chunk({ content: thinking }, 'length'))
  const model = new ChatWebLLM({ engine, modelId, thinkingPrefilled: true, request })
  const result = await model.invoke('Hello')

  expect(result.response_metadata.finish_reason).toBe('length')
  expect(result.text).toBe('')
  expect(result.content).toEqual([
    expect.objectContaining({ type: 'reasoning', reasoning: thinking }),
  ])
  expect(create).toHaveBeenCalledOnce()
})

it('keeps unfinished thinking and the limit reason in a Pi result', async () => {
  const { engine, create } = setupEngine(chunk({ content: thinking }, 'length'))
  const provider = createWebLLMProvider({
    engine,
    modelId,
    contextWindow: 4096,
    thinkingPrefilled: true,
    request,
  })
  const context = normalizeContext({ messages: [{ role: 'user', content: 'Hello', timestamp: 0 }] })
  const result = await provider.stream(provider.getModels()[0]!, context).result()

  expect(result.stopReason).toBe('length')
  expect(result.content).toEqual([expect.objectContaining({ type: 'thinking', thinking })])
  expect(create).toHaveBeenCalledOnce()
})
