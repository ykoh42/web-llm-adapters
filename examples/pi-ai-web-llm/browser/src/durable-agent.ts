import { BACKGROUND_CONTEXT as context } from '@earendil-works/chord/context'
import { Type } from '@earendil-works/pi-ai'
import {
  createRegistry,
  defineExtension,
  defineTool,
  Harness,
  type ConversationView,
} from '@earendil-works/pi-durable'
import { load, provider, unload } from './engine.js'
import { openStorage } from './storage.js'

export async function openDurable(
  modelId: string,
  render: (view: ConversationView) => void,
  onProgress: (text: string) => void,
) {
  const storage = await openStorage(`pi-webllm-v2-${modelId}`)
  let harness: Harness | undefined
  try {
    const enabled = !modelId.startsWith('Qwen2.5-')
    await load(modelId, true, onProgress)
    const { models, model } = await provider(enabled)
    const registry = createRegistry()
    registry.install(
      defineExtension({
        name: 'lookup',
        tools: [
          defineTool({
            name: 'lookup',
            description: 'Look up the answer for a code.',
            parameters: Type.Object({ code: Type.String() }),
            replay: 'safe',
            execute: async ({ code }) => ({
              output: [{ type: 'text', text: `The answer for ${code} is ORCHID-42.` }],
            }),
          }),
        ],
      }),
    )
    harness = await Harness.open(storage, { models, registry }, context)
    const root = await harness.root(context, {
      agent: {
        model: { provider: model.provider, modelId: model.id },
        instructions: 'Use lookup when asked to look up a code. Report its returned answer.',
      },
    })
    await root.configure({ thinkingLevel: enabled && model.reasoning ? 'low' : 'off' }, context)
    const watch = await root.watch(context)
    render(watch.value)
    watch.start(async (view) => render(view))
    harness.resume()
    await root.waitForIdle(context)
    return {
      run: async (prompt: string) => {
        const submission = await root.submit({ type: 'input', content: prompt }, context)
        return submission.wait(context)
      },
      cancel: () => root.abort(context),
      close: async () => {
        try {
          await harness!.close(context)
        } finally {
          await unload()
        }
      },
    }
  } catch (error) {
    await (harness ? harness.close(context) : storage.close(context)).catch(
      (cleanupError: unknown) => console.error(cleanupError),
    )
    await unload().catch((cleanupError: unknown) => console.error(cleanupError))
    throw error
  }
}
