import { provider } from './engine.js'
import type { WebLLMRequest } from 'pi-ai-web-llm'

export async function createAgent(enabled: boolean, request: WebLLMRequest = {}) {
  const [{ Agent }, { Type }, { models, model }] = await Promise.all([
    import('@earendil-works/pi-agent-core'),
    import('@earendil-works/pi-ai'),
    provider(enabled, { request }),
  ])
  return new Agent({
    initialState: {
      model,
      thinkingLevel: enabled && model.reasoning ? 'low' : 'off',
      systemPrompt: 'Use lookup when asked to look up a code. Report its returned answer.',
      tools: [
        {
          name: 'lookup',
          label: 'Lookup',
          description: 'Look up the answer for a code.',
          parameters: Type.Object({ code: Type.String() }),
          execute: async (_id, args) => ({
            content: [
              {
                type: 'text',
                text: `The answer for ${(args as { code: string }).code} is ORCHID-42.`,
              },
            ],
            details: {},
          }),
        },
      ],
    },
    streamFn: models.streamSimple.bind(models),
  })
}
