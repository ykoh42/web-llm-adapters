import { afterEach, expect, it, vi } from 'vitest'
import {
  Message,
  TextBlock,
  ModelError,
  ModelThrottledError,
  ContextWindowOverflowError,
} from '@strands-agents/sdk'
import { stampRetryable, getRetryable, ContextOverflowError } from '@langchain/core/errors'
import { BrowserModel, createBrowserBridge as strandsBridge } from 'strands-web-llm'
import {
  BrowserChatWebLLM,
  createBrowserBridge as langchainBridge,
  type BrowserPage,
} from 'langchain-web-llm'
import { chunk, collect, setupEngine } from '../../fixtures/web-llm.js'

afterEach(() => vi.unstubAllGlobals())

function open(client: 'strands' | 'langchain', error: unknown) {
  const { engine, create } = setupEngine(chunk({ content: 'Recovered' }, 'stop'))
  create.mockRejectedValueOnce(error)
  const bridge = client === 'strands' ? strandsBridge(engine) : langchainBridge(engine)
  vi.stubGlobal('window', { webllmBridge: bridge, webllmLangChainBridge: bridge })
  const page: BrowserPage = {
    async evaluate(run, arg) {
      const result = await run(structuredClone(arg))
      return result === undefined ? result : JSON.parse(JSON.stringify(result))
    },
  }
  async function invoke() {
    if (client === 'langchain')
      return new BrowserChatWebLLM(page, { modelId: 'test-model' }).invoke('Hello')
    return collect(
      new BrowserModel(page, { modelId: 'test-model' }).stream([
        new Message({ role: 'user', content: [new TextBlock('Hello')] }),
      ]),
    )
  }
  return { bridge, create, page, invoke }
}

const values = [
  undefined,
  null,
  false,
  0,
  'Worker failed',
  ['decode', 0, false],
  { step: 'decode', details: [0, false, null] },
]

for (const { client, ErrorClass } of [
  { client: 'strands', ErrorClass: ModelError },
  { client: 'strands', ErrorClass: ModelThrottledError },
  { client: 'strands', ErrorClass: ContextWindowOverflowError },
  { client: 'langchain', ErrorClass: ContextOverflowError },
] as const) {
  it.each(values.map((cause) => ({ cause })))(
    `${client} keeps ${ErrorClass.name} causes: $cause`,
    async ({ cause }) => {
      const native =
        client === 'langchain'
          ? ContextOverflowError.fromError(new Error('Native failure', { cause }))
          : Object.assign(new ErrorClass('Native failure'), { cause })
      const { bridge, invoke } = open(client, native)
      try {
        const failure = await invoke().catch((error: unknown) => error)
        expect(failure).toBeInstanceOf(ErrorClass)
        expect(failure).toHaveProperty(client === 'langchain' ? 'cause.cause' : 'cause', cause)
        expect(JSON.stringify(await invoke())).toContain('Recovered')
      } finally {
        await bridge.close()
      }
    },
  )
}

for (const client of ['strands', 'langchain'] as const) {
  it(`${client} leaves an absent cause absent`, async () => {
    const { bridge, invoke } = open(client, new Error('Native failure'))
    try {
      const failure = await invoke().catch((error: unknown) => error)
      expect(failure).toMatchObject({ name: 'Error', message: 'Native failure' })
      expect(failure).not.toHaveProperty('cause')
      expect(JSON.stringify(await invoke())).toContain('Recovered')
    } finally {
      await bridge.close()
    }
  })

  it(`${client} retains a proxy error cause across browser transport`, async () => {
    const cause = new Proxy(new RangeError('Inner failure'), {
      getPrototypeOf() {
        throw new Error('Cannot inspect provider prototype')
      },
    })
    const { bridge, invoke } = open(client, new Error('Native failure', { cause }))
    try {
      const failure = await invoke().catch((error: unknown) => error)
      expect(failure).toMatchObject({
        message: 'Native failure',
        cause: { name: 'RangeError', message: 'Inner failure' },
      })
      expect(JSON.stringify(await invoke())).toContain('Recovered')
    } finally {
      await bridge.close()
    }
  })

  it(`${client} restores a proxied context overflow as its framework error`, async () => {
    const native = Object.assign(new Error('ContextWindowSizeExceededError: input is too long.'), {
      name: 'ContextWindowSizeExceededError',
    })
    const error = new Proxy(native, {
      getPrototypeOf() {
        throw new Error('Cannot inspect provider prototype')
      },
    })
    const { bridge, invoke } = open(client, error)
    const ErrorClass = client === 'strands' ? ContextWindowOverflowError : ContextOverflowError
    try {
      const failure = await invoke().catch((error: unknown) => error)
      expect(failure).toBeInstanceOf(ErrorClass)
      expect(failure).toHaveProperty('cause.name', native.name)
      expect(failure).toHaveProperty('cause.message', native.message)
      if (client === 'langchain') expect(getRetryable(failure)).toBe(false)
      expect(JSON.stringify(await invoke())).toContain('Recovered')
    } finally {
      await bridge.close()
    }
  })

  it.each(values.flatMap((cause) => [false, true].map((nested) => ({ cause, nested }))))(
    `${client} preserves cause values through JSON transport (cause=$cause, nested=$nested)`,
    async ({ cause, nested }) => {
      const native = new TypeError('Native failure', {
        cause: nested ? new RangeError('Inner failure', { cause }) : cause,
      })
      const { bridge, create, invoke } = open(client, native)
      try {
        const failure = await invoke().catch((error: unknown) => error)
        if (!(failure instanceof Error)) throw failure
        expect(failure).toBeInstanceOf(TypeError)
        const target = nested ? failure.cause : failure
        expect(target).toHaveProperty('cause', cause)
        expect(JSON.stringify(await invoke())).toContain('Recovered')
        expect(create).toHaveBeenCalledTimes(2)
      } finally {
        await bridge.close()
      }
    },
  )
}

it.each(values.map((cause) => ({ cause })))(
  'LangChain keeps a raw native rejection as its cause: $cause',
  async ({ cause }) => {
    const { bridge, invoke } = open('langchain', cause)
    try {
      const failure = await invoke().catch((error: unknown) => error)
      expect(failure).toHaveProperty('cause', cause)
      expect(JSON.stringify(await invoke())).toContain('Recovered')
    } finally {
      await bridge.close()
    }
  },
)

it.each(['replace', 'metadata', 'clear', 'remove'] as const)(
  'LangChain applies cause edits from the final failed-attempt callback: %s',
  async (mode) => {
    const native = new Error('Native failure', { cause: 'Original cause' })
    const { bridge, create, page } = open('langchain', native)
    const changed = mode === 'metadata' ? { step: 'Changed cause' } : 'Changed cause'
    const onFailedAttempt = vi.fn((error: Error) => {
      expect(error.cause).toBe('Original cause')
      if (mode === 'remove') delete error.cause
      else error.cause = mode === 'clear' ? undefined : changed
      stampRetryable(error, false)
    })
    const model = new BrowserChatWebLLM(page, {
      modelId: 'test-model',
      maxRetries: 0,
      onFailedAttempt,
    })
    try {
      const failure = await model.invoke('Hello').catch((error: unknown) => error)
      expect(onFailedAttempt).toHaveBeenCalledOnce()
      expect(create).toHaveBeenCalledOnce()
      expect(failure).toMatchObject({ name: 'Error', message: 'Native failure' })
      expect(getRetryable(failure)).toBe(false)
      for (const error of [failure, native]) {
        if (mode === 'remove') expect(error).not.toHaveProperty('cause')
        else expect(error).toHaveProperty('cause', mode === 'clear' ? undefined : changed)
      }
      expect((await model.invoke('Recover')).text).toBe('Recovered')
    } finally {
      await bridge.close()
    }
  },
)

const cyclic: Record<string, unknown> = {}
cyclic.self = cyclic
const unsupported = [
  { name: 'cyclic object', cause: cyclic },
  { name: 'non-finite number', cause: { code: Infinity } },
  { name: 'BigInt', cause: 1n },
  {
    name: 'throwing toJSON',
    cause: {
      toJSON() {
        throw new Error('Cannot serialize cause')
      },
    },
  },
]

for (const client of ['strands', 'langchain'] as const) {
  it.each(unsupported)(
    `${client} keeps the original error when its cause is a $name`,
    async ({ cause }) => {
      const native = new TypeError('Native failure', {
        cause: new RangeError('Inner failure', { cause }),
      })
      const { bridge, invoke } = open(client, native)
      try {
        const failure = await invoke().catch((error: unknown) => error)
        expect(failure).toBeInstanceOf(TypeError)
        expect(failure).toMatchObject({
          message: 'Native failure',
          cause: { name: 'RangeError', message: 'Inner failure' },
        })
        if (!(failure instanceof Error)) throw failure
        expect(failure.cause).not.toHaveProperty('cause')
        expect(JSON.stringify(await invoke())).toContain('Recovered')
      } finally {
        await bridge.close()
      }
    },
  )
}
