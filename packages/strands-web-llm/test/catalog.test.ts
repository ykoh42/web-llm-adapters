import { ImageBlock, Message, TextBlock, ToolResultBlock, ToolUseBlock } from '@strands-agents/sdk'
import {
  functionCallingModelIds,
  ModelType,
  postInitAndCheckFieldsChatCompletion,
  postInitAndCheckFieldsCompletion,
  postInitAndCheckFieldsEmbedding,
  prebuiltAppConfig,
  type Completion,
} from '@mlc-ai/web-llm'
import { describe, expect, it, vi } from 'vitest'
import { WebLLMModel } from '../src/index.js'
import { chunk, chunks, collect, setup } from '../fixtures/engine.js'

const catalog = prebuiltAppConfig.model_list
const chatModels = catalog.filter((model) => model.model_type !== ModelType.embedding)
const visionModels = chatModels.filter((model) => model.model_type === ModelType.VLM)
const textModels = chatModels.filter((model) => model.model_type !== ModelType.VLM)
const gemma4Models = chatModels.filter((model) => model.model_id.startsWith('gemma-4-'))
const embeddingModels = catalog.filter((model) => model.model_type === ModelType.embedding)
const toolModelIds = new Set(functionCallingModelIds)
const customToolModels = new Set([
  'Qwen2.5-0.5B-Instruct',
  'Qwen2.5-1.5B-Instruct',
  'Qwen2.5-3B-Instruct',
  'Qwen2.5-7B-Instruct',
  'Qwen3-0.6B',
  'Qwen3-1.7B',
  'Qwen3-4B',
  'Qwen3-8B',
  'Qwen3.5-0.8B',
  'Qwen3.5-2B',
  'Qwen3.5-4B',
  'Qwen3.5-9B',
])
const customToolModelIds = new Set(
  chatModels
    .filter((model) => customToolModels.has(model.model_id.replace(/-q[^-]+-MLC$/, '')))
    .map((model) => model.model_id),
)
const textHistory = [
  new Message({ role: 'user', content: [new TextBlock('Hello')] }),
  new Message({ role: 'assistant', content: [new TextBlock('Hi')] }),
  new Message({ role: 'user', content: [new TextBlock('Continue')] }),
]

function chatType(model: (typeof chatModels)[number]) {
  return model.model_type === ModelType.VLM ? ModelType.VLM : ModelType.LLM
}

function checkChatRequest(request: unknown, modelId: string, modelType: ModelType) {
  postInitAndCheckFieldsChatCompletion(
    structuredClone(request) as Parameters<typeof postInitAndCheckFieldsChatCompletion>[0],
    modelId,
    modelType,
  )
}

describe('WebLLM model catalog', () => {
  it('keeps text and history requests valid for every prebuilt chat model', async () => {
    const { engine, create } = setup()
    create.mockImplementation(async () => chunks(chunk({ content: 'ok' }, 'stop')))

    for (const model of chatModels) {
      const adapter = new WebLLMModel({ engine, modelId: model.model_id })
      await collect(adapter.stream(textHistory, { systemPrompt: 'Be brief.' }))
      const request = create.mock.calls.at(-1)![0]
      expect(
        () => checkChatRequest(request, model.model_id, chatType(model)),
        model.model_id,
      ).not.toThrow()
    }
  })

  it('keeps native generation options valid for every prebuilt chat model', async () => {
    const { engine, create } = setup()
    create.mockImplementation(async () => chunks(chunk({ content: 'ok' }, 'stop')))

    for (const model of chatModels) {
      const adapter = new WebLLMModel({
        engine,
        modelId: model.model_id,
        maxTokens: 32,
        temperature: 0.25,
        topP: 0.75,
        request: {
          max_tokens: 64,
          temperature: 0.9,
          top_p: 0.1,
          frequency_penalty: 0.2,
          presence_penalty: 0.3,
          seed: 7,
          stop: ['<stop>'],
          logit_bias: { '1': 0.5 },
          logprobs: true,
          top_logprobs: 1,
          response_format: { type: 'json_object' },
          extra_body: { enable_latency_breakdown: true },
        },
      })
      await collect(adapter.stream(textHistory, { systemPrompt: 'Be brief.' }))
      const request = create.mock.calls.at(-1)![0]

      expect(request, model.model_id).toMatchObject({
        max_tokens: 32,
        temperature: 0.25,
        top_p: 0.75,
        frequency_penalty: 0.2,
        presence_penalty: 0.3,
        seed: 7,
        stop: ['<stop>'],
        logit_bias: { '1': 0.5 },
        logprobs: true,
        top_logprobs: 1,
        response_format: { type: 'json_object' },
        extra_body: { enable_latency_breakdown: true },
      })
      expect(
        () => checkChatRequest(request, model.model_id, chatType(model)),
        model.model_id,
      ).not.toThrow()
    }
  })

  it('keeps image requests valid for every prebuilt vision model', async () => {
    const { engine, create } = setup()
    create.mockImplementation(async () => chunks(chunk({ content: 'red' }, 'stop')))
    const messages = [
      new Message({
        role: 'user',
        content: [
          new TextBlock('Describe this image.'),
          new ImageBlock({ format: 'png', source: { bytes: new Uint8Array([137, 80, 78, 71]) } }),
        ],
      }),
    ]

    for (const model of visionModels) {
      const adapter = new WebLLMModel({ engine, modelId: model.model_id })
      await collect(adapter.stream(messages))
      const request = create.mock.calls.at(-1)![0]
      expect(
        () => checkChatRequest(request, model.model_id, ModelType.VLM),
        model.model_id,
      ).not.toThrow()
    }
  })

  it('lets WebLLM reject images for every prebuilt text-only model', async () => {
    const { engine, create } = setup()
    create.mockImplementation(async () => chunks(chunk({ content: 'ok' }, 'stop')))
    const messages = [
      new Message({
        role: 'user',
        content: [
          new TextBlock('Describe this image.'),
          new ImageBlock({ format: 'png', source: { bytes: new Uint8Array([137, 80, 78, 71]) } }),
        ],
      }),
    ]

    for (const model of textModels) {
      const adapter = new WebLLMModel({ engine, modelId: model.model_id })
      await collect(adapter.stream(messages))
      const request = create.mock.calls.at(-1)![0]
      expect(
        () => checkChatRequest(request, model.model_id, ModelType.LLM),
        model.model_id,
      ).toThrow(/only supports string content/)
    }
  })

  it('keeps tool history valid for every WebLLM function-calling model', async () => {
    const { engine, create } = setup()
    create.mockImplementation(async () => chunks(chunk({ content: 'ok' }, 'stop')))
    const tools = [{ name: 'clock', description: 'Get the time.' }]
    const messages = [
      textHistory[0]!,
      new Message({
        role: 'assistant',
        content: [new ToolUseBlock({ toolUseId: 'clock-1', name: 'clock', input: {} })],
      }),
      new Message({
        role: 'user',
        content: [
          new ToolResultBlock({
            toolUseId: 'clock-1',
            status: 'success',
            content: [new TextBlock('12:00')],
          }),
        ],
      }),
      new Message({ role: 'assistant', content: [new TextBlock('It is 12:00.')] }),
      new Message({ role: 'user', content: [new TextBlock('Thanks.')] }),
    ]

    for (const modelId of functionCallingModelIds) {
      const adapter = new WebLLMModel({ engine, modelId })
      await collect(adapter.stream(messages, { toolSpecs: tools, toolChoice: { auto: {} } }))
      const request = create.mock.calls.at(-1)![0]
      expect(() => checkChatRequest(request, modelId, ModelType.LLM), modelId).not.toThrow()
    }
  })

  it('keeps Gemma 4 tool prompts valid for every prebuilt Gemma 4 model', async () => {
    const { engine } = setup()
    const create = vi.spyOn(engine.completions, 'create')
    const answer = (modelId: string): Completion => ({
      id: 'test',
      object: 'text_completion',
      model: modelId,
      created: 0,
      choices: [{ text: 'ok', index: 0, finish_reason: 'stop' }],
    })

    for (const model of gemma4Models) {
      create.mockResolvedValueOnce(
        (async function* () {
          yield answer(model.model_id)
        })(),
      )
      const adapter = new WebLLMModel({ engine, modelId: model.model_id })
      await collect(
        adapter.stream(textHistory, {
          toolSpecs: [{ name: 'clock', description: 'Get the time.' }],
        }),
      )
      const request = create.mock.calls.at(-1)![0]
      expect(request.prompt).toContain('<|tool>declaration:clock')
      expect(() =>
        postInitAndCheckFieldsCompletion(structuredClone(request), model.model_id),
      ).not.toThrow()
    }
  })

  it('uses custom tool prompts only for the selected Qwen model IDs', async () => {
    const { engine, create } = setup()
    create.mockImplementation(async () => chunks(chunk({ content: 'ok' }, 'stop')))
    const tools = [{ name: 'clock', description: 'Get the time.' }]

    for (const modelId of customToolModelIds) {
      const adapter = new WebLLMModel({ engine, modelId })
      await collect(adapter.stream(textHistory, { toolSpecs: tools }))
      const request = create.mock.calls.at(-1)![0]
      expect(request, modelId).not.toHaveProperty('tools')
      const system = request.messages[0]?.content
      expect(typeof system, modelId).toBe('string')
      if (typeof system !== 'string') continue
      if (modelId.startsWith('Qwen3.5-'))
        expect(system, modelId).toContain('<function=example_function_name>')
      else if (modelId.startsWith('Qwen')) expect(system, modelId).toContain('<tool_call>')
      else expect(system, modelId).toContain('<|tool|>')
    }
  })

  it('confirms other prebuilt chat models reject adapter tool requests at WebLLM validation', async () => {
    const { engine, create } = setup()
    create.mockImplementation(async () => chunks(chunk({ content: 'ok' }, 'stop')))
    const tools = [{ name: 'clock', description: 'Get the time.' }]
    const unsupported = chatModels.filter(
      (model) =>
        !toolModelIds.has(model.model_id) &&
        !customToolModelIds.has(model.model_id) &&
        !model.model_id.startsWith('gemma-4-'),
    )

    for (const model of unsupported) {
      const adapter = new WebLLMModel({ engine, modelId: model.model_id })
      await collect(adapter.stream(textHistory, { toolSpecs: tools }))
      const request = create.mock.calls.at(-1)![0]
      expect(
        () => checkChatRequest(request, model.model_id, chatType(model)),
        model.model_id,
      ).toThrow(/not supported for ChatCompletionRequest\.tools/)
    }
  })

  it('keeps native embedding requests valid for every prebuilt embedding model', () => {
    for (const model of embeddingModels) {
      expect(
        () =>
          postInitAndCheckFieldsEmbedding(
            { model: model.model_id, input: 'hello' },
            model.model_id,
          ),
        model.model_id,
      ).not.toThrow()
    }
  })
})
