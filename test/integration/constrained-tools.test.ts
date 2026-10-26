import { expect, it } from 'vitest'
import { Message, TextBlock } from '@strands-agents/sdk'
import { HumanMessage } from '@langchain/core/messages'
import { createModels } from '@earendil-works/pi-ai/models'
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript'
import { WebLLMModel } from '../../packages/strands-web-llm/src/index.js'
import { ChatWebLLM } from '../../packages/langchain-web-llm/src/index.js'
import { createWebLLMProvider } from '../../packages/pi-ai-web-llm/src/index.js'
import { chunk, collect, setupEngine } from '../../fixtures/web-llm.js'
import type { JSONSchema } from '../../shared/web-llm/types.js'

const clients = ['strands', 'langchain', 'pi'] as const
const models = ['Qwen3-4B-q4f16_1-MLC', 'Qwen3.5-4B-q4f16_1-MLC']
const parameters = {
  type: 'object' as const,
  properties: { code: { type: 'string' as const, enum: ['001', '010'] } },
  required: ['code'],
  additionalProperties: false,
}
const spec = { name: 'lookup', description: 'Look up a code.', inputSchema: parameters }
const nativeTool = {
  type: 'function',
  function: { name: spec.name, description: spec.description, parameters },
}

for (const client of clients) {
  it.each(
    models.flatMap((modelId) =>
      ['001', '001 <think> and </tool_call>'].map((code) => ({ modelId, code })),
    ),
  )(
    `${client} $modelId keeps constrained tool calls with text mode for $code`,
    async ({ modelId, code }) => {
      const schema = {
        ...parameters,
        properties: { code: { type: 'string' as const, enum: [code, '010'] } },
      }
      const toolSpec = { ...spec, inputSchema: schema }
      const toolDefinition = {
        type: 'function',
        function: { name: spec.name, description: spec.description, parameters: schema },
      }
      const output =
        '<think>Check the code.</think>' +
        (modelId.startsWith('Qwen3.5-')
          ? `<tool_call><function=lookup><parameter=code>${code}</parameter></function></tool_call>`
          : `<tool_call>${JSON.stringify({ name: 'lookup', arguments: { code } })}</tool_call>`)
      const { engine, create } = setupEngine(chunk({ content: output }, 'stop'))
      const requests = [
        { constrainedToolCalls: false },
        { constrainedToolCalls: true },
        { constrainedToolCalls: true, response_format: { type: 'text' as const } },
      ].map((request) => ({
        ...request,
        max_tokens: 4096,
        seed: 7,
        extra_body: { enable_thinking: true },
      }))
      for (const request of requests) {
        if (client === 'strands') {
          const model = new WebLLMModel({ engine, modelId, request })
          const blocks = await collect(
            model.streamAggregated(
              [new Message({ role: 'user', content: [new TextBlock('Find ' + code + '.')] })],
              { toolSpecs: [toolSpec] },
            ),
          )
          expect(blocks).toContainEqual(
            expect.objectContaining({ type: 'reasoningBlock', text: 'Check the code.' }),
          )
          expect(blocks).toContainEqual(
            expect.objectContaining({
              type: 'toolUseBlock',
              name: 'lookup',
              input: { code },
            }),
          )
        } else if (client === 'langchain') {
          const model = new ChatWebLLM({ engine, modelId, request }).bindTools([toolDefinition])
          const response = await model.invoke([new HumanMessage('Find ' + code + '.')])
          expect(response.content).toContainEqual(
            expect.objectContaining({
              type: 'reasoning',
              reasoning: 'Check the code.',
            }),
          )
          expect(response.tool_calls).toMatchObject([{ name: 'lookup', args: { code } }])
        } else {
          const provider = createWebLLMProvider({ engine, modelId, request, contextWindow: 8192 })
          const registry = createModels()
          registry.setProvider(provider)
          const response = await registry.complete(
            provider.getModels()[0]!,
            normalizeContext({
              messages: [{ role: 'user', content: 'Find ' + code + '.', timestamp: 0 }],
              tools: [{ name: spec.name, description: spec.description, parameters: schema }],
            }),
          )
          expect(response.stopReason).toBe('toolUse')
          expect(response.content).toContainEqual({ type: 'thinking', thinking: 'Check the code.' })
          expect(response.content).toContainEqual(
            expect.objectContaining({
              type: 'toolCall',
              name: 'lookup',
              arguments: { code },
            }),
          )
        }
      }
      const baseline = create.mock.calls[0]![0]
      const constrainedCalls = create.mock.calls.slice(1).map(([request]) => {
        const { response_format, ...native } = request
        return { native, response_format }
      })
      expect(constrainedCalls).toHaveLength(2)
      for (const { native, response_format } of constrainedCalls) {
        expect(native).toEqual(baseline)
        expect(native).not.toHaveProperty('constrainedToolCalls')
        expect(response_format?.type).toBe('structural_tag')
      }
      expect(constrainedCalls[1]).toEqual(constrainedCalls[0])
      expect(schema.properties.code.enum).toEqual([code, '010'])
    },
  )
}

it.each(clients)('%s constrains type-less string and numeric enums', async (client) => {
  const schema = {
    ...parameters,
    properties: {
      code: { enum: ['001', '010'] },
      count: { enum: [7, 9] },
      payload: { const: { $ref: 'literal' } },
    },
    required: ['code', 'count', 'payload'],
  } satisfies JSONSchema
  const toolSpec = { ...spec, inputSchema: schema }
  const toolDefinition = {
    type: 'function',
    function: { name: spec.name, description: spec.description, parameters: schema },
  }
  const content =
    '<tool_call><function=lookup><parameter=code>001</parameter><parameter=count>7</parameter><parameter=payload>{"$ref":"literal"}</parameter></function></tool_call>'
  const { engine, create } = setupEngine(chunk({ content }, 'stop'))
  const request = { constrainedToolCalls: true }

  if (client === 'strands') {
    const model = new WebLLMModel({ engine, modelId: models[1]!, request })
    const blocks = await collect(
      model.streamAggregated(
        [new Message({ role: 'user', content: [new TextBlock('Find 001.')] })],
        { toolSpecs: [toolSpec] },
      ),
    )
    expect(blocks).toContainEqual(
      expect.objectContaining({
        type: 'toolUseBlock',
        name: 'lookup',
        input: { code: '001', count: 7, payload: { $ref: 'literal' } },
      }),
    )
  } else if (client === 'langchain') {
    const model = new ChatWebLLM({ engine, modelId: models[1]!, request }).bindTools([
      toolDefinition,
    ])
    const response = await model.invoke('Find 001.')
    expect(response.tool_calls).toMatchObject([
      { name: 'lookup', args: { code: '001', count: 7, payload: { $ref: 'literal' } } },
    ])
  } else {
    const provider = createWebLLMProvider({
      engine,
      modelId: models[1]!,
      request,
      contextWindow: 8192,
    })
    const registry = createModels()
    registry.setProvider(provider)
    const response = await registry.complete(
      provider.getModels()[0]!,
      normalizeContext({
        messages: [{ role: 'user', content: 'Find 001.', timestamp: 0 }],
        tools: [{ name: spec.name, description: spec.description, parameters: schema }],
      }),
    )
    expect(response.stopReason).toBe('toolUse')
    expect(response.content).toContainEqual(
      expect.objectContaining({
        type: 'toolCall',
        name: 'lookup',
        arguments: { code: '001', count: 7, payload: { $ref: 'literal' } },
      }),
    )
  }

  expect(create.mock.calls[0]![0].response_format?.type).toBe('structural_tag')
})

it.each(clients)('%s constrains an object-or-null tool argument', async (client) => {
  const schema = {
    ...parameters,
    required: ['payload'],
    properties: {
      payload: {
        anyOf: [
          {
            type: 'object',
            properties: { code: { type: 'string' } },
            required: ['code'],
            additionalProperties: false,
          },
          { type: 'null' },
        ],
      },
    },
  } satisfies JSONSchema
  const toolSpec = { ...spec, inputSchema: schema }
  const toolDefinition = {
    type: 'function',
    function: { name: spec.name, description: spec.description, parameters: schema },
  }
  const content =
    '<tool_call><function=lookup><parameter=payload>{"code":"A7"}</parameter></function></tool_call>'
  const { engine, create } = setupEngine(chunk({ content }, 'stop'))
  const request = { constrainedToolCalls: true }

  if (client === 'strands') {
    const model = new WebLLMModel({ engine, modelId: models[1]!, request })
    const blocks = await collect(
      model.streamAggregated(
        [new Message({ role: 'user', content: [new TextBlock('Find A7.')] })],
        { toolSpecs: [toolSpec] },
      ),
    )
    expect(blocks).toContainEqual(
      expect.objectContaining({
        type: 'toolUseBlock',
        name: 'lookup',
        input: { payload: { code: 'A7' } },
      }),
    )
  } else if (client === 'langchain') {
    const model = new ChatWebLLM({ engine, modelId: models[1]!, request }).bindTools([
      toolDefinition,
    ])
    const response = await model.invoke('Find A7.')
    expect(response.tool_calls).toMatchObject([
      { name: 'lookup', args: { payload: { code: 'A7' } } },
    ])
  } else {
    const provider = createWebLLMProvider({
      engine,
      modelId: models[1]!,
      request,
      contextWindow: 8192,
    })
    const registry = createModels()
    registry.setProvider(provider)
    const response = await registry.complete(
      provider.getModels()[0]!,
      normalizeContext({
        messages: [{ role: 'user', content: 'Find A7.', timestamp: 0 }],
        tools: [{ name: spec.name, description: spec.description, parameters: schema }],
      }),
    )
    expect(response.stopReason).toBe('toolUse')
    expect(response.content).toContainEqual(
      expect.objectContaining({
        type: 'toolCall',
        name: 'lookup',
        arguments: { payload: { code: 'A7' } },
      }),
    )
  }

  expect(create.mock.calls[0]![0].response_format?.type).toBe('structural_tag')
})

it.each(models)(
  '%s honors framework required choices without changing tool instructions',
  async (modelId) => {
    const output = modelId.startsWith('Qwen3.5-')
      ? '<tool_call><function=lookup><parameter=code>001</parameter></function></tool_call>'
      : '<tool_call>{"name":"lookup","arguments":{"code":"001"}}</tool_call>'
    const { engine, create } = setupEngine(chunk({ content: output }, 'stop'))
    const request = { constrainedToolCalls: true, extra_body: { enable_thinking: true } }
    const first = new WebLLMModel({ engine, modelId, request })
    await collect(
      first.stream([new Message({ role: 'user', content: [new TextBlock('Find 001.')] })], {
        toolSpecs: [spec],
        toolChoice: { any: {} },
      }),
    )
    const second = new ChatWebLLM({ engine, modelId, request })
    await second.bindTools([nativeTool], { tool_choice: 'required' }).invoke('Find 001.')
    expect(create.mock.calls[1]![0]).toEqual(create.mock.calls[0]![0])
    expect(create.mock.calls[0]![0]).not.toHaveProperty('tool_choice')
    expect(create.mock.calls[0]![0].response_format).toMatchObject({
      structural_tag: { format: { type: 'sequence' } },
    })
  },
)

it.each(clients)(
  '%s preserves literal thinking markers on a non-thinking model',
  async (client) => {
    const modelId = 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC'
    const text = 'The literal marker is <think>'
    const { engine } = setupEngine(chunk({ content: text }, 'stop'))
    const request = { constrainedToolCalls: true }
    if (client === 'strands') {
      const model = new WebLLMModel({ engine, modelId, request })
      const blocks = await collect(
        model.streamAggregated(
          [new Message({ role: 'user', content: [new TextBlock('Print the literal marker.')] })],
          { toolSpecs: [spec] },
        ),
      )
      expect(blocks.filter((block) => block.type === 'textBlock')).toEqual([new TextBlock(text)])
      expect(blocks.some((block) => block.type === 'reasoningBlock')).toBe(false)
    } else if (client === 'langchain') {
      const model = new ChatWebLLM({ engine, modelId, request }).bindTools([nativeTool])
      const response = await model.invoke('Print the literal marker.')
      expect(response.text).toBe(text)
      expect(response.content).toEqual([expect.objectContaining({ type: 'text', text })])
      expect(response.tool_calls).toEqual([])
    } else {
      const provider = createWebLLMProvider({ engine, modelId, request, contextWindow: 8192 })
      const registry = createModels()
      registry.setProvider(provider)
      const response = await registry.complete(
        provider.getModels()[0]!,
        normalizeContext({
          messages: [{ role: 'user', content: 'Print the literal marker.', timestamp: 0 }],
          tools: [{ name: spec.name, description: spec.description, parameters }],
        }),
      )
      expect(response.stopReason).toBe('stop')
      expect(response.content).toEqual([{ type: 'text', text }])
    }
  },
)
