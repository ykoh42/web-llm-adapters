import { expect, it } from 'vitest'
import {
  ModelType,
  postInitAndCheckFieldsChatCompletion,
  type ChatCompletionContentPart,
} from '@mlc-ai/web-llm'
import { formatUserContent } from '../../shared/web-llm/content.js'

const image = { type: 'image_url' as const, image_url: { url: 'https://example.org/image.png' } }
const text = (text: string) => ({ type: 'text' as const, text })

it('merges one contiguous text span and preserves its surrounding media', () => {
  const parts: ChatCompletionContentPart[] = [image, text('서울'), text(''), text(' 😀'), image]
  const before = structuredClone(parts)
  Object.freeze(parts)
  const output = formatUserContent(parts)
  expect(output).toEqual([image, text('서울 😀'), image])
  expect(output[0]).toBe(image)
  expect(output[2]).toBe(image)
  expect(parts).toEqual(before)
})

it.each([[], [text('A')], [text('A'), text('B')]])(
  'keeps text-only messages as strings %#',
  (...parts) => {
    expect(formatUserContent(parts)).toBe(parts.map((part) => part.text).join(''))
  },
)

it.each([[image], [image, text('A'), image]])(
  'returns unmerged content unchanged %#',
  (...parts) => {
    expect(formatUserContent(parts)).toBe(parts)
  },
)

it('drops empty text around media while preserving the remaining content order', () => {
  const output = formatUserContent([text(''), image, text('B'), text('')])

  expect(output).toEqual([image, text('B')])
  expect(() =>
    postInitAndCheckFieldsChatCompletion(
      { messages: [{ role: 'user', content: output }] },
      'test-model',
      ModelType.VLM,
    ),
  ).not.toThrow()
})

it('still rejects nonempty text separated by media', () => {
  expect(() => formatUserContent([text('A'), image, text('B')])).toThrow(
    'cannot preserve text around media blocks',
  )
})
