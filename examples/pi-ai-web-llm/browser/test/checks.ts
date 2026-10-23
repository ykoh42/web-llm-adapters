import { withEngine } from 'pi-ai-web-llm'
import type { AssistantMessageEvent } from '@earendil-works/pi-ai'
import { load, provider } from '../src/engine.js'
import { createAgent } from '../src/agent.js'
import { vision } from './vision.js'

async function run(prompt: string) {
  const agent = await createAgent(true)
  await agent.prompt(prompt)
  return agent.state.messages
}

async function respond(prompt: string, enabled = true) {
  const { models, model } = await provider(enabled)
  const events: AssistantMessageEvent['type'][] = []
  const stream = models.streamSimple(
    model,
    {
      messages: [{ role: 'user', content: prompt, timestamp: Date.now() }],
    },
    enabled && model.reasoning ? { reasoning: 'low' } : {},
  )
  for await (const event of stream) events.push(event.type)
  return { message: await stream.result(), events }
}

async function cancelAndRecover() {
  const { models, model } = await provider(true)
  const controller = new AbortController()
  const stream = models.streamSimple(
    model,
    {
      messages: [
        {
          role: 'user',
          content: 'Explain the history of mathematics in detail.',
          timestamp: Date.now(),
        },
      ],
    },
    { signal: controller.signal },
  )
  let observed = false
  for await (const event of stream) {
    if (!observed && (event.type === 'text_delta' || event.type === 'thinking_delta')) {
      observed = true
      controller.abort()
    }
  }
  return {
    observed,
    cancelled: await stream.result(),
    recovered: await respond('What is 3 + 3? Reply briefly.', false),
  }
}

const piChecks = { load, run, cancelAndRecover, provider, createAgent, vision, withEngine }
Object.assign(window, { piChecks })
declare global {
  interface Window {
    piChecks: typeof piChecks
  }
}
