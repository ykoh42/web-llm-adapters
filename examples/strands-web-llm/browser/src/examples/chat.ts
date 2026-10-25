import { sampling } from '../../../sampling.js'
import { Agent } from '@strands-agents/sdk'
import { WebLLMModel } from 'strands-web-llm'
import { runAgent, type ExampleInput } from './output.js'

export async function chat(input: ExampleInput) {
  const model = new WebLLMModel({
    engine: input.engine,
    modelId: input.modelId,
    thinkingPrefilled: input.modelId.startsWith('Qwen3.5-'),
    contextWindowLimit: 2048,
    maxTokens: 256,
    ...sampling(input.modelId, false),
  })
  const agent = new Agent({ model, printer: false })
  await runAgent(agent, input.prompt, input)
}

export async function json(input: ExampleInput) {
  const settings = sampling(input.modelId, false)
  let tokensWithLogprobs = 0
  const model = new WebLLMModel({
    engine: input.engine,
    modelId: input.modelId,
    thinkingPrefilled: input.modelId.startsWith('Qwen3.5-'),
    contextWindowLimit: 2048,
    maxTokens: 128,
    ...settings,
    request: {
      ...settings.request,
      seed: 42,
      logprobs: true,
      top_logprobs: 2,
      response_format: {
        type: 'json_object',
        schema: JSON.stringify({
          type: 'object',
          properties: { name: { type: 'string' }, count: { type: 'integer' } },
          required: ['name', 'count'],
          additionalProperties: false,
        }),
      },
    },
    onChunk: (chunk) => {
      tokensWithLogprobs += chunk.choices[0]?.logprobs?.content?.length ?? 0
    },
  })
  const agent = new Agent({ model, printer: false })
  const text = await runAgent(agent, `Return JSON with name and count. ${input.prompt}`, input)
  if (!input.signal.aborted) input.info({ parsed: JSON.parse(text), tokensWithLogprobs })
}

export async function grammar(input: ExampleInput) {
  const settings = sampling(input.modelId, false)
  const model = new WebLLMModel({
    engine: input.engine,
    modelId: input.modelId,
    thinkingPrefilled: input.modelId.startsWith('Qwen3.5-'),
    contextWindowLimit: 2048,
    maxTokens: 32,
    ...settings,
    request: {
      ...settings.request,
      response_format: { type: 'grammar', grammar: 'root ::= "red" | "green" | "blue"' },
    },
  })
  await runAgent(new Agent({ model, printer: false }), input.prompt, input)
}
