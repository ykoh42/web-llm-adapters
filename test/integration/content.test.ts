import { expect, it } from 'vitest'
import type { ChatCompletionContentPart } from '@mlc-ai/web-llm'
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

it.each([[], [text('A'), text('B')]])('keeps text-only messages as strings %#', (...parts) => {
  expect(formatUserContent(parts)).toBe(parts.map((part) => part.text).join(''))
})

it.each([[image], [image, text('A'), image]])(
  'returns unmerged content unchanged %#',
  (...parts) => {
    expect(formatUserContent(parts)).toBe(parts)
  },
)

it('rejects text separated by media even when a text block is empty', () => {
  expect(() => formatUserContent([text(''), image, text('B')])).toThrow(
    'cannot preserve text around media blocks',
  )
})
