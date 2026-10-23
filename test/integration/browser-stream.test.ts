import { afterEach, expect, it, vi } from 'vitest'
import { browserStream } from '../../shared/browser/stream.js'

afterEach(() => vi.unstubAllGlobals())

function pause() {
  let resume!: () => void
  const wait = new Promise<void>((resolve) => (resume = resolve))
  return { wait, resume }
}

const turn = () => new Promise<void>((resolve) => setImmediate(resolve))

async function* forward<T>(source: AsyncGenerator<T>): AsyncGenerator<T> {
  yield* source
}

it('bundles native chunks with parsed output without reading ahead', async () => {
  vi.stubGlobal('scheduler', { yield: turn })
  const output = browserStream<number, string>([])
  let reads = 0
  const source = forward(
    forward(
      (async function* () {
        reads++
        await turn()
        output.push('raw')
        yield 1
        reads++
        yield 2
      })(),
    ),
  )
  try {
    expect(await output.next(source)).toEqual({ done: false, value: 1, chunks: ['raw'] })
    expect(reads).toBe(1)
  } finally {
    await source.return(undefined)
  }
})

it.each([
  { name: 'a later task', scheduler: { yield: turn } },
  { name: 'an aborted task', scheduler: { yield: () => Promise.reject(new Error('Aborted')) } },
  { name: 'an unsupported scheduler', scheduler: undefined },
])('delivers native chunks before parsing finishes with $name', async ({ scheduler }) => {
  vi.stubGlobal('scheduler', scheduler)
  const output = browserStream<number, string>([])
  const hold = pause()
  const source = (async function* () {
    await turn()
    output.push('first')
    output.push('second')
    await hold.wait
    yield 1
  })()
  try {
    expect(await output.next(source)).toEqual({
      native: true,
      done: false,
      chunks: ['first', 'second'],
    })
    hold.resume()
    expect(await output.next(source)).toEqual({ done: false, value: 1 })
  } finally {
    hold.resume()
    await source.return(undefined)
  }
})

it('publishes ready output without waiting for the scheduled notification', async () => {
  const task = pause()
  vi.stubGlobal('scheduler', { yield: () => task.wait })
  const output = browserStream<number, string>([])
  const source = (async function* () {
    await turn()
    output.push('raw')
    yield 1
  })()
  try {
    expect(await output.next(source)).toEqual({ done: false, value: 1, chunks: ['raw'] })
  } finally {
    task.resume()
    await source.return(undefined)
  }
})

it('delivers already buffered native chunks without waiting for parsed output', async () => {
  const task = pause()
  const hold = pause()
  vi.stubGlobal('scheduler', { yield: () => task.wait })
  const output = browserStream<number, string>([])
  output.push('buffered')
  const source = (async function* () {
    await hold.wait
    yield 1
  })()
  try {
    expect(await output.next(source)).toEqual({ native: true, done: false, chunks: ['buffered'] })
  } finally {
    task.resume()
    hold.resume()
    await source.return(undefined)
  }
})

it('does not let an old notification wake a later read', async () => {
  const task = pause()
  const hold = pause()
  vi.stubGlobal('scheduler', { yield: () => task.wait })
  const output = browserStream<number, string>([])
  const source = (async function* () {
    await turn()
    output.push('raw')
    yield 1
    await hold.wait
    yield 2
  })()
  try {
    expect(await output.next(source)).toEqual({ done: false, value: 1, chunks: ['raw'] })
    const pending = output.next(source)
    let completed = false
    void pending.then(() => (completed = true))
    task.resume()
    await turn()
    expect(completed).toBe(false)
    hold.resume()
    expect(await pending).toEqual({ done: false, value: 2 })
  } finally {
    task.resume()
    hold.resume()
    await source.return(undefined)
  }
})

it('schedules one notification for each native batch without losing chunks', async () => {
  const schedule = vi.fn(turn)
  vi.stubGlobal('scheduler', { yield: schedule })
  const output = browserStream<number, number>([])
  const second = pause()
  const finish = pause()
  const source = (async function* () {
    await turn()
    for (let index = 0; index < 1000; index++) output.push(index)
    await second.wait
    for (let index = 1000; index < 2000; index++) output.push(index)
    await finish.wait
    yield 1
  })()
  try {
    expect(await output.next(source)).toEqual({
      native: true,
      done: false,
      chunks: Array.from({ length: 1000 }, (_, index) => index),
    })
    expect(schedule).toHaveBeenCalledTimes(1)
    const pending = output.next(source)
    second.resume()
    expect(await pending).toEqual({
      native: true,
      done: false,
      chunks: Array.from({ length: 1000 }, (_, index) => index + 1000),
    })
    expect(schedule).toHaveBeenCalledTimes(2)
    finish.resume()
    expect(await output.next(source)).toEqual({ done: false, value: 1 })
    expect(await output.next(source)).toEqual({ done: true, value: undefined })
  } finally {
    second.resume()
    finish.resume()
    await source.return(undefined)
  }
})
