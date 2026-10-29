import { expect, it } from 'vitest'
import { Type } from '@earendil-works/pi-ai'
import { createModels } from '@earendil-works/pi-ai/models'
import { WebLLMModel } from '../../packages/strands-web-llm/src/index.js'
import { ChatWebLLM } from '../../packages/langchain-web-llm/src/index.js'
import { createWebLLMProvider } from '../../packages/pi-ai-web-llm/src/index.js'
import { nativeChunks, nativeHistory } from '../../shared/web-llm/native.js'
import { chunk, chunks, collect, setupEngine } from '../../fixtures/web-llm.js'

const modelId = 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC'
const raw = '[{"name":"lookup","arguments":{"value":-0,"nested":[0,-0.0]}}]'
// WebLLM parses and reserializes arguments before returning tool_calls.
const input = JSON.stringify(JSON.parse(raw)[0].arguments)
const call = { index: 0, function: { name: 'lookup', arguments: input } }
const schema = {
  type: 'object' as const,
  properties: { value: { type: 'number' as const }, nested: { type: 'array' as const } },
}

function check(value: unknown) {
  const args = value as { value: number; nested: number[] }
  expect(Object.is(args.value, -0)).toBe(true)
  expect(Object.is(args.nested[0], 0)).toBe(true)
  expect(Object.is(args.nested[1], -0)).toBe(true)
}

it('preserves native negative zero in all three frameworks', async () => {
  const { engine } = setupEngine(
    chunk({ content: raw }),
    chunk({ tool_calls: [call] }, 'tool_calls'),
  )
  const tool = { name: 'lookup', description: '', inputSchema: schema }
  const strands = new WebLLMModel({ engine, modelId })
  const events = await collect(strands.stream([], { toolSpecs: [tool] }))
  const argumentsText = events
    .filter((event) => event.type === 'modelContentBlockDeltaEvent')
    .map((event) => (event.delta.type === 'toolUseInputDelta' ? event.delta.input : ''))
    .join('')
  check(JSON.parse(argumentsText))

  const langchain = new ChatWebLLM({ engine, modelId })
  const answer = await langchain
    .bindTools([{ type: 'function', function: { name: 'lookup', parameters: schema } }])
    .invoke('Look up the value.')
  expect(answer.text).toBe('')
  check(answer.tool_calls?.[0]?.args)

  const provider = createWebLLMProvider({ engine, modelId, contextWindow: 8192 })
  const models = createModels()
  models.setProvider(provider)
  const result = await models.complete(provider.getModels()[0]!, {
    messages: [{ role: 'user', content: 'Look up the value.', timestamp: 0 }],
    tools: [
      {
        name: 'lookup',
        description: '',
        parameters: Type.Object({ value: Type.Number(), nested: Type.Array(Type.Number()) }),
      },
    ],
  })
  const toolCall = result.content.find((part) => part.type === 'toolCall')
  expect(toolCall?.type).toBe('toolCall')
  if (toolCall?.type !== 'toolCall') throw new Error('Missing tool call.')
  check(toolCall.arguments)
})

it('does not complete a call whose changed arguments were already streamed', async () => {
  const output = nativeChunks(
    chunks(chunk({ content: raw }), chunk({ tool_calls: [call] }), chunk({}, 'tool_calls')),
    true,
  )
  await expect(collect(output)).rejects.toThrow('changed negative zero')
})

it('keeps the native chunks unchanged for observers', async () => {
  const native: string[] = []
  const output = await collect(
    nativeChunks(
      chunks(chunk({ content: raw }), chunk({ tool_calls: [call] }, 'tool_calls')),
      true,
      {
        onChunk: (chunk) => {
          const argumentsText = chunk.choices[0]?.delta.tool_calls?.[0]?.function?.arguments
          if (argumentsText) native.push(argumentsText)
        },
      },
    ),
  )
  expect(native).toEqual([input])
  check(JSON.parse(output.flatMap((part) => part.calls)[0]!.input!))
})

it('does not treat different zero signs as the same native history', () => {
  const messages = nativeHistory(
    [
      {
        role: 'assistant',
        content: '[{"name":"lookup","arguments":{"value":0}}]',
        tool_calls: [
          { id: 'call', type: 'function', function: { name: 'lookup', arguments: '{"value":-0}' } },
        ],
      },
    ],
    modelId,
  )
  const message = messages[0]
  expect(message?.role).toBe('assistant')
  if (message?.role !== 'assistant') throw new Error('Missing assistant message.')
  expect(message.content).toContain('{"value":-0}')
})
