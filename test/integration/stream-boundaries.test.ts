import { describe, expect, it } from 'vitest'
import type { Completion } from '@mlc-ai/web-llm'
import { AIMessageChunk } from '@langchain/core/messages'
import { gemmaChunks } from '../../shared/web-llm/gemma/stream.js'
import { qwenChunks } from '../../shared/web-llm/qwen/stream.js'
import { qwen35Chunks } from '../../shared/web-llm/qwen35/stream.js'
import type { StreamChunk } from '../../shared/web-llm/stream/chunks.js'
import type { ToolSpec } from '../../shared/web-llm/types.js'
import { readStream } from '../../packages/strands-web-llm/src/stream/read.js'
import { readChunks } from '../../packages/langchain-web-llm/src/chat/chunks.js'
import { responseStream } from '../../packages/pi-ai-web-llm/src/events.js'
import { chunk } from '../../fixtures/web-llm.js'

type Finish = 'stop' | 'length' | 'abort' | null
type Framework = 'Strands' | 'LangChain' | 'Pi'
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
      properties: {
        text: { type: 'string' },
        count: { type: 'integer' },
        payload: { type: 'object' },
        items: { type: 'array' },
      },
    },
  },
]
const text = '서울 😀 \\ "quoted"\nline <think> {nested}'
const gemmaCall = `<|tool_call>call:lookup{count:7,text:<|"|>${text}<|"|>}<tool_call|>`
const jsonCall = String.raw`<tool_call>{"name":"lookup","arguments":{"text":"literal </tool_call>, <tool_call>, and \\\"quote\"","count":3}}</tool_call>`
const xmlCall = `<tool_call><function=lookup><parameter=text>\n${text}\n</parameter><parameter=count>\n7\n</parameter></function></tool_call>`
const xmlEndText = 'literal </tool_call>, then more'
const xmlEndPayload = { nested: ['</tool_call>', { '</tool_call>': 'keep' }] }
const xmlEndCall =
  `<tool_call><function=lookup><parameter=text>\n${xmlEndText}\n</parameter>` +
  `<parameter=payload>\n${JSON.stringify(xmlEndPayload)}\n</parameter></function></tool_call>`
const xmlFunctionText = 'literal </function>, </tool_call>, and <function=lookup> stay here'
const xmlFunctionPayload = { nested: ['</function>', { '<function=lookup>': '</tool_call>' }] }
const xmlFunctionCall =
  `<tool_call><function=lookup><parameter=text>\n${xmlFunctionText}\n</parameter>` +
  `<parameter=payload>\n${JSON.stringify(xmlFunctionPayload)}\n</parameter></function></tool_call>`
const xmlJsonPayload = {
  nested: ['<parameter=x>', '</parameter>', '</function>', '</tool_call>', '서울 😀 \\ "quote"'],
  '</parameter>': { '<function=lookup>': '<parameter=payload>' },
}
const xmlJsonCall =
  '<tool_call><function=lookup>' +
  `<parameter=payload>\n${JSON.stringify(xmlJsonPayload)}\n</parameter>` +
  `<parameter=items>\n${JSON.stringify(xmlJsonPayload.nested)}\n</parameter>` +
  '</function></tool_call>'
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
  {
    name: 'Qwen XML literal closing tags',
    source: `Before<think>Plan 서울 😀</think>Answer${xmlEndCall}${xmlEndCall}After<tool_`,
    text: 'BeforeAnswerAfter<tool_',
    calls: Array.from({ length: 2 }, () => ({
      name: 'lookup',
      input: JSON.stringify({ text: xmlEndText, payload: xmlEndPayload }),
    })),
    parse: (parts, finish, failure) => qwen35Chunks(chat(parts, finish, failure), tools),
  },
  {
    name: 'Qwen XML literal function tags',
    source: `Before<think>Plan 서울 😀</think>Answer${xmlFunctionCall}`,
    text: 'BeforeAnswer',
    calls: [
      {
        name: 'lookup',
        input: JSON.stringify({ text: xmlFunctionText, payload: xmlFunctionPayload }),
      },
    ],
    parse: (parts, finish, failure) => qwen35Chunks(chat(parts, finish, failure), tools),
  },
  {
    name: 'Qwen XML JSON parameter tags',
    source: `Before<think>Plan 서울 😀</think>Answer${xmlJsonCall}${xmlJsonCall}After<tool_`,
    text: 'BeforeAnswerAfter<tool_',
    calls: Array.from({ length: 2 }, () => ({
      name: 'lookup',
      input: JSON.stringify({ payload: xmlJsonPayload, items: xmlJsonPayload.nested }),
    })),
    parse: (parts, finish, failure) => qwen35Chunks(chat(parts, finish, failure), tools),
  },
]
const scenarios = (['stop', 'length', 'abort', null] as const).flatMap((finish) =>
  [false, true].map((partial) => ({ finish, partial })),
)

describe.each(protocols)('$name chunk boundaries', (protocol) => {
  describe.each(['Strands', 'LangChain', 'Pi'] as const)('%s reader', (framework) => {
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
            stop: framework === 'LangChain' ? 'tool_calls' : 'toolUse',
            error: undefined,
          })
        } else expect(expected.calls).toEqual([])
        if (failure) {
          expect(expected.error).toHaveProperty('message', failure.message)
          if (framework !== 'Pi') expect(expected.error).toBe(failure)
        }
        if (finish === 'stop' && partial) expect(expected.error).toBeInstanceOf(Error)

        for (const parts of partitions(source)) {
          const actual = await read(framework, protocol.parse(parts, finish, failure))
          expect(actual, `chunk lengths: ${parts.map((part) => part.length).join(',')}`).toEqual(
            expected,
          )
          if (expected.error instanceof Error)
            expect(actual.error).toHaveProperty('name', expected.error.name)
          if (failure && framework !== 'Pi') expect(actual.error).toBe(failure)
        }
      },
    )
  })
})

describe.each([
  {
    name: 'Gemma',
    thought: '<|channel>thought\nPlan 서울 😀<channel|>',
    call: gemmaCall,
    invalid: gemmaCall.replace('call:lookup', 'call:missing'),
    parse: (parts: string[]) => gemmaChunks(raw(parts, 'stop'), tools),
  },
  {
    name: 'Qwen JSON',
    thought: '<think>Plan 서울 😀</think>',
    call: jsonCall,
    invalid: jsonCall.replace('"name":"lookup"', '"name":"missing"'),
    parse: (parts: string[]) => qwenChunks(chat(parts, 'stop'), tools),
  },
  {
    name: 'Qwen XML',
    thought: '<think>Plan 서울 😀</think>',
    call: xmlCall,
    invalid: xmlCall.replace('<function=lookup>', '<function=missing>'),
    parse: (parts: string[]) => qwen35Chunks(chat(parts, 'stop'), tools),
  },
])('$name invalid second tool call', ({ name, thought, call, invalid, parse }) => {
  it.each(['Strands', 'LangChain', 'Pi'] as const)(
    'keeps received output without publishing tools through %s',
    async (framework) => {
      const source = `Before${thought}Answer${call}${invalid}`
      for (const parts of partitions(source)) {
        const result = await read(framework, parse(parts))
        expect(result, `chunk lengths: ${parts.map((part) => part.length).join(',')}`).toEqual({
          text: name === 'Gemma' ? `BeforeAnswer${call}${invalid}` : 'BeforeAnswer',
          reasoning: 'Plan 서울 😀',
          calls: [],
          stop: framework === 'Pi' ? 'error' : undefined,
          error: expect.objectContaining({ message: expect.stringMatching(/unknown.*tool/i) }),
        })
      }
    },
  )
})

describe.each([
  {
    name: 'Qwen JSON',
    call: jsonCall,
    invalid: jsonCall.replace('"name":"lookup"', '"name":"missing"'),
    parse: qwenChunks,
  },
  {
    name: 'Qwen XML',
    call: xmlCall,
    invalid: xmlCall.replace('<function=lookup>', '<function=missing>'),
    parse: qwen35Chunks,
  },
])('$name failed tool frame', ({ call, invalid, parse }) => {
  it.each(['Strands', 'LangChain', 'Pi'] as const)(
    'keeps the already-received text and thinking through %s',
    async (framework) => {
      const source = `Before${invalid}After<think>Later</think>End${call}Final`
      const result = await read(framework, parse(chat([source], 'stop'), tools))
      expect(result).toEqual({
        text: 'BeforeAfterEndFinal',
        reasoning: 'Later',
        calls: [],
        stop: framework === 'Pi' ? 'error' : undefined,
        error: expect.objectContaining({ message: expect.stringMatching(/unknown.*tool/i) }),
      })
    },
  )
})

describe.each([
  { name: 'Qwen JSON', parse: qwenChunks },
  { name: 'Qwen XML', parse: qwen35Chunks },
])('$name unclosed thinking', ({ parse }) => {
  describe.each(['Strands', 'LangChain', 'Pi'] as const)('%s reader', (framework) => {
    it.each([
      { finish: 'stop', stopRequested: false },
      { finish: 'stop', stopRequested: true },
      { finish: 'length', stopRequested: false },
      { finish: 'abort', stopRequested: false },
    ] as const)('preserves $finish output with stopRequested=$stopRequested', async (test) => {
      const thought = 'Plan 서울 😀</thi'
      const source = 'Before<think>' + thought
      const run = (parts: string[]) => read(framework, parse(chat(parts, test.finish), tools, test))
      const expected = await run([source])
      expect(expected.text).toBe('Before')
      expect(expected.reasoning).toBe(thought)
      expect(expected.calls).toEqual([])
      if (test.finish === 'stop' && !test.stopRequested) {
        expect(expected.error).toHaveProperty('message', 'Qwen ended inside a thinking block.')
        expect(expected.stop).toBe(framework === 'Pi' ? 'error' : undefined)
      } else if (test.finish === 'abort') {
        if (framework === 'Strands') {
          expect(expected.error).toBeUndefined()
          expect(expected.stop).toBe('cancelled')
        } else {
          expect(expected.error).toBeInstanceOf(Error)
          expect(expected.stop).toBe(framework === 'Pi' ? 'aborted' : undefined)
        }
      } else {
        expect(expected.error).toBeUndefined()
        const stops = {
          Strands: { stop: 'endTurn', length: 'maxTokens' },
          LangChain: { stop: 'stop', length: 'length' },
          Pi: { stop: 'stop', length: 'length' },
        }
        expect(expected.stop).toBe(stops[framework][test.finish])
      }
      for (const parts of partitions(source)) expect(await run(parts)).toEqual(expected)
    })
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

describe.each(['Strands', 'LangChain', 'Pi'] as const)('%s stream completion', (framework) => {
  it.each([
    { name: 'text', value: { text: 'Late' } },
    { name: 'thinking', value: { reasoning: 'Late' } },
    { name: 'another stop', value: { stop: 'endTurn' as const } },
    {
      name: 'tool calls',
      value: { calls: [{ index: 0, id: 'late-call', name: 'lookup', input: '{}' }] },
    },
  ])('rejects $name received after the stop reason', async ({ value }) => {
    const empty: StreamChunk = {
      text: '',
      reasoning: '',
      calls: [],
      stop: undefined,
      usage: undefined,
    }
    const result = await read(
      framework,
      (async function* () {
        yield { ...empty, text: 'Done', stop: 'endTurn' as const }
        yield { ...empty, ...value }
      })(),
    )

    expect(result.text).toBe('Done')
    expect(result.reasoning).toBe('')
    expect(result.calls).toEqual([])
    expect(result.error).toHaveProperty('message', 'WebLLM returned content after its stop reason.')
  })
})

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
    } else if (framework === 'LangChain') {
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
    } else {
      const stream = responseStream(
        {
          id: 'test-model',
          name: 'Test model',
          api: 'webllm',
          provider: 'webllm',
          baseUrl: '',
          input: ['text'],
          reasoning: true,
          contextWindow: 8192,
          maxTokens: 4096,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        },
        () => ({ chunks, toolNames: ['lookup'] }),
      )
      for await (const event of stream) {
        if (event.type === 'text_delta') result.text += event.delta
        if (event.type === 'thinking_delta') result.reasoning += event.delta
        if (event.type === 'toolcall_end')
          result.calls.push({
            name: event.toolCall.name,
            input: JSON.stringify(event.toolCall.arguments),
          })
        if (event.type === 'done') result.stop = event.reason
        if (event.type === 'error') {
          result.stop = event.reason
          result.error = new Error(event.error.errorMessage)
        }
      }
    }
  } catch (error) {
    result.error = error
  }
  return result
}
