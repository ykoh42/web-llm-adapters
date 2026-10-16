import { afterEach, expect, it, vi } from 'vitest'
import { Message, TextBlock, type ModelStreamEvent } from '@strands-agents/sdk'
import type { Page } from '@playwright/test'
import { BrowserModel } from '../browser-model.js'
import { SummaryModel } from '../summary-model.js'

afterEach(() => vi.restoreAllMocks())

const model = () => new SummaryModel({} as Page, { modelId: 'gemma-4-E2B-it-q4f16_1-MLC' })
const content = (text: string) => [
  new Message({
    role: 'user',
    content: [new TextBlock('<content>'), new TextBlock(text), new TextBlock('</content>')],
  }),
]

async function* answer(text: string): AsyncGenerator<ModelStreamEvent> {
  yield { type: 'modelContentBlockDeltaEvent', delta: { type: 'textDelta', text } }
  yield { type: 'modelMessageStopEvent', stopReason: 'endTurn' }
}

async function text(stream: AsyncIterable<ModelStreamEvent>) {
  let text = ''
  for await (const event of stream)
    if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta')
      text += event.delta.text
  return text
}

it('keeps every word intact and combines actual partial summaries with the same instructions', async () => {
  const source = 'note '.repeat(798) + 'build-한글-🦉-12345678 ' + 'note '.repeat(500)
  const calls = vi.spyOn(BrowserModel.prototype, 'stream')
  calls.mockImplementationOnce(() => answer('First fact.'))
  calls.mockImplementationOnce(() => answer('Second fact.'))
  calls.mockImplementationOnce(() => answer('Combined facts.'))
  const options = { systemPrompt: 'Keep facts.', cancelSignal: new AbortController().signal }
  expect(await text(model().stream(content(source), options))).toBe('Combined facts.')
  const parts = calls.mock.calls.slice(0, -1).map(([messages]) => {
    expect(messages[0]?.content[0]).toEqual(new TextBlock('<content>'))
    expect(messages[0]?.content.at(-1)).toEqual(new TextBlock('</content>'))
    const block = messages[0]?.content[1]
    if (block?.type !== 'textBlock') throw new Error('Missing summary input')
    return block.text
  })
  expect(parts.join('')).toBe(source)
  expect(parts.some((part) => part.includes('build-한글-🦉-12345678'))).toBe(true)
  expect(calls.mock.calls.at(-1)?.[0][0]?.content).toEqual(
    content('First fact.\n\nSecond fact.')[0]!.content,
  )
  for (const [, forwarded] of calls.mock.calls) expect(forwarded).toBe(options)
})

it('leaves short summaries and ordinary fetched-page requests unchanged', async () => {
  const calls = vi.spyOn(BrowserModel.prototype, 'stream').mockImplementation(() => answer('OK'))
  const short = content('One fact.')
  const fetched = [new Message({ role: 'user', content: [new TextBlock('page '.repeat(1000))] })]
  await text(model().stream(short))
  await text(model().stream(fetched))
  expect(calls.mock.calls.map(([messages]) => messages)).toEqual([short, fetched])
})

it.each(['empty', 'truncated', 'error'])(
  'does not combine a failed %s partial summary',
  async (failure) => {
    const calls = vi.spyOn(BrowserModel.prototype, 'stream').mockImplementation(async function* () {
      if (failure === 'error') throw new Error('Cancelled or failed')
      if (failure === 'empty') yield* answer('')
      else yield { type: 'modelMessageStopEvent', stopReason: 'maxTokens' }
    })
    await expect(text(model().stream(content('note '.repeat(1000))))).rejects.toThrow()
    expect(calls).toHaveBeenCalledTimes(1)
  },
)
