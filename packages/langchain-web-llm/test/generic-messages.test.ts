import { afterEach, expect, it, vi } from 'vitest'
import {
  AIMessage,
  ChatMessage,
  ChatMessageChunk,
  HumanMessage,
  SystemMessage,
  ToolMessage,
  type BaseMessage,
} from '@langchain/core/messages'
import {
  BrowserChatWebLLM,
  ChatWebLLM,
  createBrowserBridge,
  type BrowserPage,
} from '../src/index.js'
import { chunk, setup } from '../fixtures/engine.js'

afterEach(() => vi.unstubAllGlobals())

const fields = {
  content: [{ type: 'text', text: 'Hello' }],
  name: 'speaker',
  id: 'message-id',
  additional_kwargs: { reasoning_content: 'A previous thought.' },
  response_metadata: { source: 'previous-model' },
}
const call = { type: 'tool_call', id: 'lookup-1', name: 'lookup', args: { code: 'A7' } }
function legacyToolMessage() {
  return {
    contentBlocks: [{ type: 'text' as const, text: 'Looking up A7.' }],
    response_metadata: { output_version: 'v1' },
    additional_kwargs: {
      tool_calls: [
        {
          id: 'lookup-1',
          type: 'function' as const,
          function: { name: 'lookup', arguments: '{"code":"A7"}' },
        },
      ],
    },
  }
}
const histories: { name: string; generic: BaseMessage[]; canonical: BaseMessage[] }[] = [
  {
    name: 'LangChain role aliases',
    generic: [
      new ChatMessage('Answer simply.', 'system'),
      new ChatMessage('Hi', 'human'),
      new ChatMessage('Hello', 'ai'),
    ],
    canonical: [
      new SystemMessage('Answer simply.'),
      new HumanMessage('Hi'),
      new AIMessage('Hello'),
    ],
  },
  {
    name: 'LangChain role aliases in chunks',
    generic: [new ChatMessageChunk('Hi', 'human'), new ChatMessageChunk('Hello', 'ai')],
    canonical: [new HumanMessage('Hi'), new AIMessage('Hello')],
  },
  {
    name: 'generic message chunks',
    generic: [new ChatMessageChunk({ role: 'user', content: 'Hello' })],
    canonical: [new HumanMessage('Hello')],
  },
  {
    name: 'typed PCM audio',
    generic: [
      new ChatMessage({
        role: 'user',
        content: [
          {
            type: 'input_audio',
            input_audio: {
              format: 'pcm_f32',
              data: new Float32Array([0, 0.25, -0.25]),
              sample_rate: 16000,
            },
          },
        ],
      }),
    ],
    canonical: [
      new HumanMessage({
        content: [
          {
            type: 'input_audio',
            input_audio: {
              format: 'pcm_f32',
              data: new Float32Array([0, 0.25, -0.25]),
              sample_rate: 16000,
            },
          },
        ],
      }),
    ],
  },

  {
    name: 'system, user and assistant text',
    generic: [
      new ChatMessage('Answer simply.', 'system'),
      new ChatMessage('Hi', 'user'),
      new ChatMessage('Hello', 'assistant'),
    ],
    canonical: [
      new SystemMessage('Answer simply.'),
      new HumanMessage('Hi'),
      new AIMessage('Hello'),
    ],
  },
  {
    name: 'user content and participant name',
    generic: [new ChatMessage({ ...fields, role: 'user' })],
    canonical: [new HumanMessage(fields)],
  },
  {
    name: 'assistant reasoning and participant name',
    generic: [new HumanMessage('Hi'), new ChatMessage({ ...fields, role: 'assistant' })],
    canonical: [new HumanMessage('Hi'), new AIMessage(fields)],
  },
  {
    name: 'assistant tools and their results',
    generic: [
      new HumanMessage('Look up A7.'),
      new ChatMessage({ role: 'assistant', content: [call] }),
      new ToolMessage({ content: '42', tool_call_id: 'lookup-1', name: 'lookup' }),
    ],
    canonical: [
      new HumanMessage('Look up A7.'),
      new AIMessage({ content: [call] }),
      new ToolMessage({ content: '42', tool_call_id: 'lookup-1', name: 'lookup' }),
    ],
  },
  {
    name: 'standard image content',
    generic: [
      new ChatMessage({
        role: 'user',
        content: [
          { type: 'text', text: 'Read this.' },
          { type: 'image_url', image_url: { url: 'https://example.com/image.png' } },
        ],
      }),
    ],
    canonical: [
      new HumanMessage({
        content: [
          { type: 'text', text: 'Read this.' },
          { type: 'image_url', image_url: { url: 'https://example.com/image.png' } },
        ],
      }),
    ],
  },
]

function client(browser: boolean) {
  const { engine, create } = setup(chunk({ content: '6' }, 'stop'))
  const bridge = createBrowserBridge(engine)
  vi.stubGlobal('window', { webllmLangChainBridge: bridge })
  const page: BrowserPage = {
    async evaluate(run, arg) {
      const value = await run(structuredClone(arg))
      return value === undefined ? value : JSON.parse(JSON.stringify(value))
    },
  }
  const model = browser
    ? new BrowserChatWebLLM(page, { modelId: 'test-model' })
    : new ChatWebLLM({ engine, modelId: 'test-model' })
  return { model, bridge, create }
}

for (const browser of [false, true]) {
  it(`preserves v1 content with legacy tool calls (browser: ${browser})`, async () => {
    const { model, bridge, create } = client(browser)
    const generic = new ChatMessage({ ...legacyToolMessage(), role: 'assistant' })
    const canonical = new AIMessage(legacyToolMessage())
    const before = structuredClone(generic.toDict())
    const start = new HumanMessage('Look up A7.')
    const result = new ToolMessage({ content: '42', tool_call_id: 'lookup-1', name: 'lookup' })
    const end = new HumanMessage('3 + 3?')
    try {
      const expected = await model.invoke([start, canonical, result, end])
      const received = await model.invoke([start, generic, result, end])
      expect(create.mock.calls[1]?.[0]).toEqual(create.mock.calls[0]?.[0])
      expect(received.content).toEqual(expected.content)
      expect(generic.toDict()).toEqual(before)
    } finally {
      await bridge.close()
    }
  })
  it.each(histories)(
    `keeps $name with generic messages (browser: ${browser})`,
    async ({ generic, canonical }) => {
      const { model, bridge, create } = client(browser)
      const before = structuredClone(generic.map((message) => message.toDict()))
      try {
        const expected = await model.invoke([...canonical, new HumanMessage('3 + 3?')])
        const received = await model.invoke([...generic, new HumanMessage('3 + 3?')])
        expect(create.mock.calls[1]?.[0]).toEqual(create.mock.calls[0]?.[0])
        expect(received.content).toEqual(expected.content)
        expect(generic.map((message) => message.toDict())).toEqual(before)
      } finally {
        await bridge.close()
      }
    },
  )
  it.each(['custom', 'developer', 'tool', '__proto__', 'constructor'])(
    `rejects unsupported generic role %s (browser: ${browser})`,
    async (role) => {
      const { model, bridge, create } = client(browser)
      try {
        await expect(model.invoke([new ChatMessage('Hi', role)])).rejects.toThrow('generic')
        expect(create).not.toHaveBeenCalled()
        expect((await model.invoke('Hi')).text).toBe('6')
      } finally {
        await bridge.close()
      }
    },
  )

  it.each([
    {
      name: 'named system messages',
      messages: [new ChatMessage({ role: 'system', content: 'Hi', name: 'speaker' })],
      error: 'cannot preserve names',
    },
    {
      name: 'late system messages',
      messages: [new HumanMessage('Hi'), new ChatMessage('Late', 'system')],
      error: 'before the conversation',
    },
    {
      name: 'signed reasoning',
      messages: [
        new ChatMessage({
          role: 'assistant',
          content: [{ type: 'reasoning', reasoning: 'Private', signature: 'signed' }],
        }),
      ],
      error: 'unsigned text reasoning',
    },
  ])(`keeps validation for $name (browser: ${browser})`, async ({ messages, error }) => {
    const { model, bridge, create } = client(browser)
    try {
      await expect(model.invoke(messages)).rejects.toThrow(error)
      expect(create).not.toHaveBeenCalled()
    } finally {
      await bridge.close()
    }
  })
}
