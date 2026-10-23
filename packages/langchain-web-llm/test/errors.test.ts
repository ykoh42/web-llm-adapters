import { describe, expect, it } from 'vitest'
import { ContextOverflowError, getRetryable } from '@langchain/core/errors'
import { AIMessage, HumanMessage } from '@langchain/core/messages'
import { createDeepAgent, createSummarizationMiddleware, StateBackend } from 'deepagents'
import { ChatWebLLM } from '../src/index.js'
import { chunk, setup } from '../fixtures/engine.js'

const message = 'ContextWindowSizeExceededError: input has 2000 tokens but the limit is 1024.'
const failures = [
  Object.assign(new Error(message), { name: 'ContextWindowSizeExceededError' }),
  message,
  new Error(message),
]

describe('LangChain errors', () => {
  it.each([undefined, null, 0, 'Worker failed'])(
    'preserves a non-Error rejection as its cause: %s',
    async (error) => {
      const { model, create } = setup()
      create.mockRejectedValueOnce(error)
      const failure = await model.invoke('Hi').catch((error: unknown) => error)
      expect(failure).toBeInstanceOf(Error)
      expect(failure).toHaveProperty('message', String(error))
      expect(failure).toHaveProperty('cause', error)
    },
  )

  it.each(failures.map((error) => ({ error })))(
    'marks an opening overflow as non-retryable: $error',
    async ({ error }) => {
      const { engine, create } = setup()
      create.mockRejectedValue(error)
      const model = new ChatWebLLM({ engine, modelId: 'test-model', maxRetries: 1 })
      const failure = await model.invoke('Hi').catch((error: unknown) => error)
      expect(ContextOverflowError.isInstance(failure)).toBe(true)
      expect(getRetryable(failure)).toBe(false)
      expect(failure).toHaveProperty('cause')
      expect(create).toHaveBeenCalledOnce()
    },
  )

  it.each(failures.map((error) => ({ error })))(
    'maps a generation overflow and releases the engine: $error',
    async ({ error }) => {
      const { model, create } = setup(chunk({ content: 'Recovered' }, 'stop'))
      create.mockResolvedValueOnce(
        (async function* () {
          yield chunk({ content: 'Started' })
          throw error
        })(),
      )
      await expect(model.invoke('Hi')).rejects.toBeInstanceOf(ContextOverflowError)
      expect((await model.invoke('After')).text).toBe('Recovered')
    },
  )

  it('lets Deep Agents summarization recover from a native context overflow', async () => {
    const { engine, create } = setup(chunk({ content: 'Summary and answer.' }, 'stop'))
    create.mockRejectedValueOnce(failures[0])
    const model = new ChatWebLLM({ engine, modelId: 'test-model' })
    const agent = createDeepAgent({
      model,
      middleware: [
        createSummarizationMiddleware({
          model,
          backend: new StateBackend(),
          trigger: { type: 'messages', value: 100 },
          keep: { type: 'messages', value: 2 },
        }),
      ],
    })
    const result = await agent.invoke({
      messages: [
        new HumanMessage('Remember code ORCHID-42.'),
        new AIMessage('I will remember it.'),
        new HumanMessage('The code is for a flower.'),
        new AIMessage('Understood.'),
        new HumanMessage('What was the code?'),
      ],
    })
    expect(result.messages.at(-1)?.text).toBe('Summary and answer.')
    expect(Object.keys(result.files ?? {})).toContainEqual(
      expect.stringMatching(/^\/conversation_history\//),
    )
    expect(JSON.stringify(result.files)).toContain('ORCHID-42')
    expect(create.mock.calls.length).toBeGreaterThanOrEqual(3)
    expect(JSON.stringify(create.mock.calls.slice(1))).toContain('conversation_history')
  })
})
