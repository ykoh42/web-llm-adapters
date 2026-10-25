import { describe, expect, it, vi } from 'vitest'
import { Message, TextBlock } from '@strands-agents/sdk'
import { createModels } from '@earendil-works/pi-ai/models'
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript'
import { WebLLMModel } from '../../packages/strands-web-llm/src/index.js'
import { ChatWebLLM } from '../../packages/langchain-web-llm/src/index.js'
import { createWebLLMProvider } from '../../packages/pi-ai-web-llm/src/index.js'
import type { WebLLMRequest } from '../../shared/web-llm/request.js'
import { chunk, setupEngine } from '../../fixtures/web-llm.js'

const prefix = '<think>\n\n</think>\n\n'
const jsonText = ' \n{"text":"<think>literal</think>","tool":"<tool_call>"}\n '
function grammar(text: string) {
  return { type: 'grammar' as const, grammar: `root ::= ${JSON.stringify(text)}` }
}
const formats = [{ type: 'json_object' as const, schema: '{}' }, grammar(jsonText)]
const clients = ['Strands', 'LangChain', 'Pi'] as const

async function read(
  client: (typeof clients)[number],
  modelId: string,
  parts: string[],
  request: WebLLMRequest,
) {
  const items = parts.map((content, index) =>
    chunk({ content }, index === parts.length - 1 ? 'stop' : null),
  )
  const { engine, create } = setupEngine(...items)
  const onChunk = vi.fn()
  const options = { engine, modelId, request, onChunk }
  let text = ''
  if (client === 'Strands') {
    const model = new WebLLMModel(options)
    for await (const event of model.stream([
      new Message({ role: 'user', content: [new TextBlock('Reply in the requested format.')] }),
    ]))
      if (event.type === 'modelContentBlockDeltaEvent') {
        expect(event.delta.type).toBe('textDelta')
        if (event.delta.type === 'textDelta') text += event.delta.text
      }
  } else if (client === 'LangChain') {
    for await (const message of await new ChatWebLLM(options).stream(
      'Reply in the requested format.',
    )) {
      expect(message.contentBlocks.every((block) => block.type === 'text')).toBe(true)
      text += message.text
    }
  } else {
    const models = createModels()
    const provider = createWebLLMProvider({ ...options, contextWindow: 4096 })
    models.setProvider(provider)
    const stream = models.stream(
      provider.getModels()[0]!,
      normalizeContext({
        messages: [{ role: 'user', content: 'Reply in the requested format.', timestamp: 0 }],
      }),
    )
    for await (const event of stream) if (event.type === 'text_delta') text += event.delta
    const result = await stream.result()
    expect(result.stopReason).toBe('stop')
    expect(result.content.every((block) => block.type === 'text')).toBe(true)
  }
  expect(onChunk.mock.calls.map(([item]) => item)).toEqual(items)
  expect(create.mock.calls[0]?.[0].response_format).toEqual(request.response_format)
  expect(create.mock.calls[0]?.[0].extra_body?.enable_thinking).toBe(
    request.extra_body?.enable_thinking,
  )
  return text
}

describe.each(clients)('%s structured Qwen output', (client) => {
  describe.each(['Qwen3-1.7B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC'])('%s', (modelId) => {
    it.each(formats)('removes only the injected prefix for $type', async (response_format) => {
      const body = jsonText
      const source = prefix + body
      for (let at = 0; at <= source.length; at++) {
        expect(
          await read(client, modelId, [source.slice(0, at), source.slice(at)], {
            response_format,
            extra_body: { enable_thinking: false },
          }),
        ).toBe(body)
      }
    })

    it.each([undefined, null, true])(
      'keeps literal leading tags when thinking is %s',
      async (enabled) => {
        const body = prefix + 'literal grammar output'
        expect(
          await read(client, modelId, [body], {
            response_format: grammar(body),
            extra_body: { ...(enabled !== undefined && { enable_thinking: enabled }) },
          }),
        ).toBe(body)
      },
    )

    it.each([prefix + 'literal output', prefix.slice(0, -1), ''])(
      'keeps the complete grammar body: %j',
      async (body) => {
        expect(
          await read(client, modelId, [prefix + body], {
            response_format: grammar(body),
            extra_body: { enable_thinking: false },
          }),
        ).toBe(body)
      },
    )

    it.each([
      '{"answer":42}',
      ' <think>\n\n</think>\n\nliteral',
      '<think>different</think>literal',
    ])('preserves an absent or different prefix: %j', async (body) => {
      expect(
        await read(client, modelId, [...body], {
          response_format: grammar(body),
          extra_body: { enable_thinking: false },
        }),
      ).toBe(body)
    })
  })
})
