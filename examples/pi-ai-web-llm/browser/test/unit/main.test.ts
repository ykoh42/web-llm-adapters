import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({ createAgent: vi.fn(), load: vi.fn() }))
vi.mock('../../src/agent.js', () => ({ createAgent: mocks.createAgent }))
vi.mock('../../src/engine.js', () => ({ load: mocks.load }))

interface Element {
  id: string
  value: string
  checked: boolean
  disabled: boolean
  textContent: string
  onclick?: () => void
  onsubmit?: (event: { preventDefault(): void }) => void
  replaceChildren(): void
}

const elements = new Map<string, Element>()
function element(selector: string) {
  let value = elements.get(selector)
  if (!value) {
    value = {
      id: selector.slice(1),
      value: selector === '#message' ? 'Look up A7.' : 'model-id',
      checked: true,
      disabled: false,
      textContent: '',
      replaceChildren() {},
    }
    elements.set(selector, value)
  }
  return value
}

beforeEach(async () => {
  vi.resetModules()
  vi.resetAllMocks()
  elements.clear()
  vi.stubGlobal('document', {
    querySelector: element,
    querySelectorAll: () => ['#load', '#send', '#cancel'].map(element),
    createTextNode: () => ({ data: '', appendData() {} }),
  })
  await import('../../src/main.js')
})
afterEach(() => vi.unstubAllGlobals())

it('cancels before the agent is ready without starting model work', async () => {
  const agent = {
    abort: vi.fn(),
    prompt: vi.fn(),
    subscribe: vi.fn(),
    state: { messages: [{ role: 'assistant', stopReason: 'stop' }] },
  }
  let ready!: () => void
  const pending = new Promise<void>((resolve) => (ready = resolve))
  mocks.createAgent.mockImplementation(async () => {
    await pending
    return agent
  })
  element('#chat').onsubmit!({ preventDefault() {} })
  try {
    await vi.waitFor(() => expect(mocks.createAgent).toHaveBeenCalledOnce())
    expect(element('#cancel').disabled).toBe(false)
    element('#cancel').onclick!()
    ready()
    await vi.waitFor(() => expect(element('#cancel').disabled).toBe(true))
    expect(agent.prompt).not.toHaveBeenCalled()
    expect(element('#status').textContent).toBe('aborted')
  } finally {
    ready()
  }
})

it('keeps chat cancellation disabled while loading a model', async () => {
  let ready!: () => void
  mocks.load.mockReturnValue(new Promise<void>((resolve) => (ready = resolve)))
  element('#load').onclick!()
  try {
    expect(element('#cancel').disabled).toBe(true)
  } finally {
    ready()
    await vi.waitFor(() => expect(element('#status').textContent).toBe('Ready'))
  }
})

it.each([false, true])('preserves a running prompt (cancel: %s)', async (cancelled) => {
  let finish!: () => void
  const pending = new Promise<void>((resolve) => (finish = resolve))
  const agent = {
    abort: vi.fn(() => finish()),
    prompt: vi.fn(() => pending),
    subscribe: vi.fn(),
    state: { messages: [{ role: 'assistant', stopReason: cancelled ? 'aborted' : 'stop' }] },
  }
  mocks.createAgent.mockResolvedValue(agent)
  element('#chat').onsubmit!({ preventDefault() {} })
  try {
    await vi.waitFor(() => expect(agent.prompt).toHaveBeenCalledWith('Look up A7.'))
    expect(mocks.createAgent).toHaveBeenCalledWith(true)
    if (cancelled) element('#cancel').onclick!()
    else finish()
    await vi.waitFor(() => expect(element('#cancel').disabled).toBe(true))
    expect(agent.abort).toHaveBeenCalledTimes(cancelled ? 1 : 0)
    expect(element('#status').textContent).toBe(cancelled ? 'aborted' : 'stop')
  } finally {
    finish()
  }
})
