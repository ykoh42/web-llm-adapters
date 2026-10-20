import { describe, expect, it } from 'vitest'
import type { Completion } from '@mlc-ai/web-llm'
import { AIMessageChunk } from '@langchain/core/messages'
import { gemmaChunks } from '../../shared/web-llm/gemma/stream.js'
import { qwenChunks } from '../../shared/web-llm/qwen/stream.js'
import { qwen35Chunks } from '../../shared/web-llm/qwen35/stream.js'
import type { StreamChunk } from '../../shared/web-llm/stream/chunks.js'
import type { ToolSpec } from '../../shared/web-llm/types.js'
import { readStream } from '../../packages/strands-web-llm/src/stream/read.js'
import { readChunks } from '../../packages/langchain-web-llm/src/chat/stream.js'
import { chunk } from '../../packages/langchain-web-llm/fixtures/engine.js'

type Finish = 'stop' | 'length' | 'abort' | null
type Framework = 'Strands' | 'LangChain'
interface Call {
  name: string
  input: string
}
interface Protocol {
  name: string
  source: string
  text: string
  calls: Call[]
  parse: (parts: string[], finish: Finish, failure?: Error) => AsyncIterable<StreamChunk>
}
interface Result {
  text: string
  reasoning: string
  calls: Call[]
  stop: string | undefined
  error: unknown
}

const tools: ToolSpec[] = [
  {
    name: 'lookup',
    description: 'Read text.',
    inputSchema: {
      type: 'object',
      properties: { text: { type: 'string' }, count: { type: 'integer' } },
    },
  },
]
const text = '서울 😀 \\ "quoted"\nline <think> {nested}'
const gemmaCall = `<|tool_call>call:lookup{count:7,text:<|"|>${text}<|"|>}<tool_call|>`
const jsonCall = String.raw`<tool_call>{"name":"lookup","arguments":{"text":"literal </tool_call>, <tool_call>, and \\\"quote\"","count":3}}</tool_call>`
const xmlCall = `<tool_call><function=lookup><parameter=text>\n${text}\n</parameter><parameter=count>\n7\n</parameter></function></tool_call>`
const protocols: Protocol[] = [
  {
    name: 'Gemma',
    source: `Before<|channel>thought\nPlan 서울 😀<channel|>Answer${gemmaCall}`,
    text: 'BeforeAnswer',
    calls: [{ name: 'lookup', input: JSON.stringify({ count: 7, text }) }],
    parse: (parts, finish, failure) => gemmaChunks(raw(parts, finish, failure), tools),
  },
  {
    name: 'Gemma prefix',
    source: `Plan 서울 😀<channel|>Answer${gemmaCall}`,
    text: 'Answer',
    calls: [{ name: 'lookup', input: JSON.stringify({ count: 7, text }) }],
    parse: (parts, finish, failure) =>
      gemmaChunks(raw(parts, finish, failure), tools, { prefix: '<|channel>thought\n' }),
  },
  {
    name: 'Qwen JSON',
    source: `Before<think>Plan 서울 😀</think>Answer${jsonCall}${jsonCall}After<tool_`,
    text: 'BeforeAnswerAfter<tool_',
    calls: Array.from({ length: 2 }, () => ({
      name: 'lookup',
      input: JSON.stringify({ text: 'literal </tool_call>, <tool_call>, and \\"quote"', count: 3 }),
    })),
    parse: (parts, finish, failure) => qwenChunks(chat(parts, finish, failure), tools),
  },
  {
    name: 'Qwen XML',
    source: `Before<think>Plan 서울 😀</think>Answer${xmlCall}${xmlCall}After<tool_`,
    text: 'BeforeAnswerAfter<tool_',
    calls: Array.from({ length: 2 }, () => ({
      name: 'lookup',
      input: JSON.stringify({ text, count: 7 }),
    })),
    parse: (parts, finish, failure) => qwen35Chunks(chat(parts, finish, failure), tools),
  },
]
const scenarios = (['stop', 'length', 'abort', null] as const).flatMap((finish) =>
  [false, true].map((partial) => ({ finish, partial })),
)

describe.each(protocols)('$name chunk boundaries', (protocol) => {
  describe.each(['Strands', 'LangChain'] as const)('%s reader', (framework) => {
    it.each(scenarios)(
      'preserves output on $finish, partial=$partial',
      async ({ finish, partial }) => {
        const source = partial
          ? protocol.source.slice(0, Math.floor(protocol.source.length * 0.6))
          : protocol.source
        const failure = finish === null ? new Error('Native stream failed.') : undefined
        const expected = await read(framework, protocol.parse([source], finish, failure))
        if (finish === 'stop' && !partial) {
          expect(expected).toEqual({
            text: protocol.text,
            reasoning: 'Plan 서울 😀',
            calls: protocol.calls,
            stop: framework === 'Strands' ? 'toolUse' : 'tool_calls',
            error: undefined,
          })
        } else expect(expected.calls).toEqual([])
        if (failure) expect(expected.error).toBe(failure)
        if (finish === 'stop' && partial) expect(expected.error).toBeInstanceOf(Error)

        for (const parts of partitions(source)) {
          const actual = await read(framework, protocol.parse(parts, finish, failure))
          expect(actual, `chunk lengths: ${parts.map((part) => part.length).join(',')}`).toEqual(
            expected,
          )
          if (expected.error instanceof Error)
            expect(actual.error).toHaveProperty('name', expected.error.name)
          if (failure) expect(actual.error).toBe(failure)
        }
      },
    )
  })
})

function* partitions(source: string): Generator<string[]> {
  for (let index = 0; index <= source.length; index++)
    yield [source.slice(0, index), source.slice(index)]
  yield source.split('')
  const parts: string[] = []
  for (let index = 0; index < source.length; index += 7) parts.push(source.slice(index, index + 7))
  yield parts
}

async function* raw(parts: string[], finish: Finish, failure?: Error): AsyncGenerator<Completion> {
  for (const [index, text] of parts.entries())
    yield {
      id: 'test',
      object: 'text_completion',
      created: 0,
      model: 'test-model',
      choices: [
        {
          index: 0,
          text,
          logprobs: null,
          finish_reason: index === parts.length - 1 ? finish : null,
        },
      ],
    }
  if (failure) throw failure
}

async function* chat(parts: string[], finish: Finish, failure?: Error) {
  for (const [index, content] of parts.entries())
    yield chunk({ content }, index === parts.length - 1 ? finish : null)
  if (failure) throw failure
}

async function read(framework: Framework, chunks: AsyncIterable<StreamChunk>): Promise<Result> {
  const result: Result = { text: '', reasoning: '', calls: [], stop: undefined, error: undefined }
  try {
    if (framework === 'Strands') {
      for await (const event of readStream(chunks)) {
        if (event.type === 'modelContentBlockStartEvent' && event.start?.type === 'toolUseStart')
          result.calls.push({ name: event.start.name, input: '' })
        if (event.type === 'modelContentBlockDeltaEvent') {
          if (event.delta.type === 'textDelta') result.text += event.delta.text
          if (event.delta.type === 'reasoningContentDelta') result.reasoning += event.delta.text
          if (event.delta.type === 'toolUseInputDelta')
            result.calls.at(-1)!.input += event.delta.input
        }
        if (event.type === 'modelMessageStopEvent') result.stop = event.stopReason
      }
    } else {
      for await (const part of readChunks(chunks, 'test-model')) {
        if (!AIMessageChunk.isInstance(part.message))
          throw new Error('Expected an AI message chunk.')
        result.text += part.text
        if (Array.isArray(part.message.content))
          for (const block of part.message.content)
            if (block.type === 'reasoning' && typeof block.reasoning === 'string')
              result.reasoning += block.reasoning
        for (const call of part.message.tool_call_chunks ?? [])
          result.calls.push({ name: call.name!, input: call.args! })
        const finish = part.message.response_metadata.finish_reason
        if (typeof finish === 'string') result.stop = finish
      }
    }
  } catch (error) {
    result.error = error
  }
  return result
}
