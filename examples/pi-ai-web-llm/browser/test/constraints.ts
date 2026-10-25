import { Agent } from '@earendil-works/pi-agent-core'
import { Type } from '@earendil-works/pi-ai'
import { createModels } from '@earendil-works/pi-ai/models'
import { createWebLLMProvider, type WebLLMChunk } from 'pi-ai-web-llm'
import { sampling } from '../../sampling.js'
import { provider } from '../src/engine.js'

export async function constrainedAgent() {
  const results = []
  for (const constrainedToolCalls of [false, true]) {
    const { engine, model: loaded } = await provider(true)
    const chunks: WebLLMChunk[] = []
    const source = createWebLLMProvider({
      engine,
      modelId: loaded.id,
      contextWindow: loaded.contextWindow,
      request: {
        ...sampling(loaded.id, loaded.reasoning),
        max_tokens: 4096,
        seed: 7,
        constrainedToolCalls,
      },
      onChunk: (chunk) => chunks.push(chunk),
    })
    const models = createModels()
    models.setProvider(source)
    const model = source.getModels()[0]!
    const inputs: unknown[] = []
    const receipt = 'CODE-' + crypto.randomUUID().slice(0, 8)
    const agent = new Agent({
      initialState: {
        model,
        thinkingLevel: model.reasoning ? 'low' : 'off',
        tools: [
          {
            name: 'lookup',
            label: 'Lookup',
            description: 'Look up a code.',
            parameters: Type.Object(
              { code: Type.String(), enabled: Type.Boolean() },
              { additionalProperties: false },
            ),
            execute: async (_id, input) => {
              inputs.push(input)
              return { content: [{ type: 'text', text: receipt }], details: {} }
            },
          },
        ],
      },
      streamFn: models.streamSimple.bind(models),
    })
    await agent.prompt(
      'Call lookup with code A7 and enabled false. After its result, reply with the returned answer.',
    )
    results.push({
      constrainedToolCalls,
      inputs,
      receipt,
      messages: agent.state.messages,
      chunks,
      error: agent.state.errorMessage,
    })
  }
  return results
}
