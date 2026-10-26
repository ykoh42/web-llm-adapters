import { expect, it } from 'vitest'
import { AIMessage, HumanMessage, ToolMessage, type BaseMessage } from '@langchain/core/messages'
import { scrubUnsupportedMultimodalContent } from 'deepagents'
import {
  functionCallingModelIds,
  ModelType,
  postInitAndCheckFieldsChatCompletion,
  prebuiltAppConfig,
} from '@mlc-ai/web-llm'
import { ChatWebLLM } from '../src/index.js'
import { formatMessages, nativeMessages } from '../src/chat/messages.js'
import { setup } from '../fixtures/engine.js'

it('allows custom tool profiles', () => {
  const { engine } = setup()
  const model = new ChatWebLLM({
    engine,
    modelId: 'Qwen3-1.7B-custom',
    profile: { toolCalling: true },
  })
  expect(model.profile.toolCalling).toBe(true)
})

it('reports unsupported mid-conversation capabilities', () => {
  const { engine } = setup()
  expect(new ChatWebLLM({ engine, modelId: 'test' }).profile).toMatchObject({
    midConversationSystemMessages: false,
    midConversationTools: false,
  })
})

it.each([
  ['midConversationSystemMessages', 'mid-conversation system messages'],
  ['midConversationTools', 'add tools mid-conversation'],
] as const)('rejects an unsupported %s profile override', (capability, message) => {
  const { engine } = setup()
  expect(
    () =>
      new ChatWebLLM({
        engine,
        modelId: 'test',
        profile: { [capability]: true },
      }),
  ).toThrow(message)
})

it.each(functionCallingModelIds)('reports native %s tool calling', (modelId) => {
  const { engine } = setup()
  expect(new ChatWebLLM({ engine, modelId }).profile).toMatchObject({
    toolCalling: true,
    toolChoice: true,
  })
})

it.each([
  'gemma-4-E2B-it-q4f16_1-MLC',
  'Qwen2.5-3B-Instruct-q4f16_1-MLC',
  'Qwen3-4B-q4f16_1-MLC',
  'Qwen3.5-4B-q4f16_1-MLC',
])('reports %s tool choice support', (modelId) => {
  const { engine } = setup()
  expect(new ChatWebLLM({ engine, modelId }).profile).toMatchObject({
    toolCalling: true,
    toolChoice: true,
  })
})

it.each([
  { profile: { toolCalling: true }, toolChoice: true },
  { profile: { toolCalling: true, toolChoice: false }, toolChoice: false },
])('uses custom tool choice capability: $toolChoice', ({ profile, toolChoice }) => {
  const { engine } = setup()
  expect(new ChatWebLLM({ engine, modelId: 'custom-model', profile }).profile).toMatchObject({
    toolCalling: true,
    toolChoice,
  })
})

it.each(['Phi-3.5-vision-instruct-q4f16_1-MLC', 'Phi-3.5-vision-instruct-q4f32_1-MLC'])(
  'reports %s vision capabilities',
  (modelId) => {
    const { engine } = setup()
    expect(new ChatWebLLM({ engine, modelId }).profile).toMatchObject({
      imageInputs: true,
      imageUrlInputs: true,
      imageToolMessage: false,
      audioInputs: false,
      toolCalling: false,
      toolChoice: false,
      reasoningOutput: false,
      structuredOutput: true,
    })
  },
)

it('profiles every vision model in the WebLLM catalog', () => {
  const { engine } = setup()
  const visionIds = prebuiltAppConfig.model_list
    .filter((record) => record.model_type === ModelType.VLM)
    .map((record) => record.model_id)

  expect(visionIds.length).toBeGreaterThan(0)
  for (const modelId of visionIds)
    expect(new ChatWebLLM({ engine, modelId }).profile.imageInputs).toBe(true)
})

it.each([
  'gemma-4-E2B-it-q4f16_1-MLC',
  'gemma-4-other',
  'Qwen3.5-4B-q4f16_1-MLC',
  ...functionCallingModelIds,
])('keeps %s audio opt-in for agent profiles', (modelId) => {
  const { engine } = setup()
  expect(new ChatWebLLM({ engine, modelId }).profile.audioInputs).toBe(false)
  expect(
    new ChatWebLLM({ engine, modelId, profile: { audioInputs: true } }).profile.audioInputs,
  ).toBe(true)
})

it.each(['gemma-4-E2B-it-q4f16_1-MLC', ...functionCallingModelIds])(
  'lets Deep Agents exclude audio that %s tool mode cannot read',
  (modelId) => {
    const { engine } = setup()
    const model = new ChatWebLLM({ engine, modelId })
    const call = new AIMessage({
      content: '',
      tool_calls: [{ id: 'audio', name: 'read_file', args: { file_path: '/sound.wav' } }],
    })
    const content = [{ type: 'audio', data: 'AQ==', mimeType: 'audio/wav' }]
    const result = new ToolMessage({ tool_call_id: 'audio', content })
    const messages = scrubUnsupportedMultimodalContent([call, result], model)

    expect(result.content).toEqual(content)
    expect(messages[1]?.text).toContain('/sound.wav')
    expect(messages[1]?.contentBlocks.every((block) => block.type === 'text')).toBe(true)
    expect(() => formatMessages(messages)).not.toThrow()
  },
)

it('keeps custom model audio support unknown unless the caller sets it', () => {
  const { engine } = setup()
  const modelId = 'custom-model'
  expect(new ChatWebLLM({ engine, modelId }).profile.audioInputs).toBeUndefined()
  for (const audioInputs of [false, true])
    expect(new ChatWebLLM({ engine, modelId, profile: { audioInputs } }).profile.audioInputs).toBe(
      audioInputs,
    )
})

it.each(functionCallingModelIds)('reports %s image limits to Deep Agents', (modelId) => {
  const { engine } = setup()
  const model = new ChatWebLLM({ engine, modelId })
  const record = prebuiltAppConfig.model_list.find((record) => record.model_id === modelId)
  expect(record).toBeDefined()
  const modelType = record!.model_type ?? ModelType.LLM
  expect(modelType).toBe(ModelType.LLM)
  const content = [
    { type: 'text', text: 'Describe this image.' },
    { type: 'image', url: 'https://example.com/image.png' },
  ]
  const original = new HumanMessage({ content })
  const validate = (messages: BaseMessage[]) =>
    postInitAndCheckFieldsChatCompletion(
      { messages: nativeMessages(formatMessages(messages)), model: modelId, stream: true },
      modelId,
      modelType,
    )

  expect(() => validate([original])).toThrow()
  const messages = scrubUnsupportedMultimodalContent([original], model)
  expect(original.content).toEqual(content)
  expect(messages[0]?.contentBlocks.every((block) => block.type === 'text')).toBe(true)
  expect(() => validate(messages)).not.toThrow()
  expect(model.profile).toMatchObject({ imageInputs: false, imageUrlInputs: false })
  expect(
    new ChatWebLLM({
      engine,
      modelId,
      profile: { imageInputs: true, imageUrlInputs: true },
    }).profile,
  ).toMatchObject({ imageInputs: true, imageUrlInputs: true })
})

it('leaves custom image capabilities to the caller', () => {
  const { engine } = setup()
  expect(new ChatWebLLM({ engine, modelId: 'custom-model' }).profile.imageInputs).toBeUndefined()
  expect(new ChatWebLLM({ engine, modelId: 'custom-model' }).profile.imageUrlInputs).toBeUndefined()
})
