import { Agent } from '@strands-agents/sdk'
import { WebLLMModel } from 'strands-web-llm'
import { runAgent, type ExampleInput } from './output.js'
import { sampling as modelSampling } from '../../../sampling.js'

export async function tags(input: ExampleInput) {
  const model = new WebLLMModel({
    engine: input.engine,
    modelId: input.modelId,
    contextWindowLimit: 2048,
    maxTokens: 96,
    temperature: 0,
    request: {
      response_format: {
        type: 'structural_tag',
        structural_tag: {
          type: 'structural_tag',
          format: {
            type: 'triggered_tags',
            triggers: ['<answer>'],
            tags: [
              {
                begin: '<answer>',
                end: '</answer>',
                content: {
                  type: 'json_schema',
                  json_schema: {
                    type: 'object',
                    properties: { count: { type: 'integer' } },
                    required: ['count'],
                    additionalProperties: false,
                  },
                },
              },
            ],
            at_least_one: true,
            stop_after_first: true,
          },
        },
      },
    },
  })
  await runAgent(new Agent({ model, printer: false }), input.prompt, input)
}

export async function thinking(input: ExampleInput) {
  const modes = []
  for (const enabled of [false, true]) {
    const model = new WebLLMModel({
      engine: input.engine,
      modelId: input.modelId,
      contextWindowLimit: 2048,
      maxTokens: 512,
      ...modelSampling(input.modelId, enabled),
    })
    const agent = new Agent({ model, printer: false })
    const text = await runAgent(agent, input.prompt, input)
    const reasoning = agent.messages
      .flatMap((message) => message.content)
      .filter((block) => block.type === 'reasoningBlock')
      .map((block) => block.text ?? '')
      .join('')
    modes.push({ enabled, text, reasoning })
    input.write('\n\n')
  }
  input.info({ thinking: modes })
}

export async function sampling(input: ExampleInput) {
  const model = new WebLLMModel({
    engine: input.engine,
    modelId: input.modelId,
    contextWindowLimit: 2048,
    maxTokens: 128,
    temperature: 0.8,
    topP: 0.9,
    request: {
      seed: 7,
      frequency_penalty: 0.3,
      presence_penalty: 0.2,
      repetition_penalty: 1.1,
      stop: ['\n'],
      logit_bias: { '0': -100 },
      logprobs: true,
      top_logprobs: 2,
      extra_body: { enable_latency_breakdown: true },
    },
  })
  const first = await runAgent(new Agent({ model, printer: false }), input.prompt, input)
  input.write('\n\n')
  const second = await runAgent(new Agent({ model, printer: false }), input.prompt, input)
  input.write('\n\n')
  const stoppedModel = new WebLLMModel({
    engine: input.engine,
    modelId: input.modelId,
    maxTokens: 32,
    request: {
      stop: ['\n'],
      response_format: { type: 'grammar', grammar: 'root ::= "alpha\\nbeta"' },
    },
  })
  const stopped = await runAgent(
    new Agent({ model: stoppedModel, printer: false }),
    'Follow the grammar.',
    input,
  )
  input.info({ first, second, sameSeedMatches: first === second, stopped })
}
