import type { Model } from '@earendil-works/pi-ai'
import type { createModels } from '@earendil-works/pi-ai/models'
import type {
  ChatCompletionChunk,
  ChatCompletionRequestStreaming,
  MLCEngineInterface,
} from '@mlc-ai/web-llm'
import { withEngine } from 'pi-ai-web-llm'

type Models = ReturnType<typeof createModels>

/** Compare identical requests on the same engine, alternating measurement order. */
export async function comparePerformance(
  models: Models,
  model: Model<'webllm'>,
  engine: MLCEngineInterface,
) {
  const context = {
    messages: [
      {
        role: 'user' as const,
        content: 'Write a short story about a sailor finding an island.',
        timestamp: 0,
      },
    ],
  }
  let request: ChatCompletionRequestStreaming | undefined
  let firstRequest: string | undefined

  async function adapter(observe = false) {
    await withEngine(engine, (engine) => engine.resetChat())
    const start = performance.now()
    let raw = ''
    const response = await models.complete(model, context, {
      onPayload(payload) {
        request = structuredClone(payload) as ChatCompletionRequestStreaming
        const serialized = JSON.stringify(request)
        if (firstRequest !== undefined && serialized !== firstRequest)
          throw new Error('The adapter changed the request during the comparison.')
        firstRequest = serialized
      },
      ...(observe && {
        onProviderStreamEvent(chunk) {
          raw += (chunk as ChatCompletionChunk).choices[0]?.delta.content ?? ''
        },
      }),
    })
    const elapsed = performance.now() - start
    if (response.stopReason !== 'stop' && response.stopReason !== 'length')
      throw new Error(response.errorMessage ?? `Unexpected stop: ${response.stopReason}`)
    const text = response.content
      .filter((part) => part.type === 'text')
      .map((part) => part.text)
      .join('')
    if (observe && text !== raw) throw new Error('The adapter changed the native text.')
    return {
      elapsed,
      tokens: response.usage.output,
      text,
    }
  }

  async function native() {
    const input = request
    if (!input) throw new Error('The adapter did not provide a native request.')
    return withEngine(engine, async (engine) => {
      await engine.resetChat()
      const start = performance.now()
      let text = ''
      let tokens = 0
      const stream = await engine.chat.completions.create(structuredClone(input))
      for await (const chunk of stream) {
        text += chunk.choices[0]?.delta.content ?? ''
        if (chunk.usage) tokens = chunk.usage.completion_tokens
      }
      return { elapsed: performance.now() - start, tokens, text }
    })
  }

  await adapter(true)
  await native()
  const samples = []
  for (const first of ['native', 'adapter', 'native', 'adapter'] as const) {
    const before = first === 'native' ? await native() : undefined
    const adapted = await adapter()
    const original = before ?? (await native())
    samples.push({ first, native: original, adapter: adapted })
  }
  return samples
}
