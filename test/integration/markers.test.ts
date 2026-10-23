import { expect, it } from 'vitest'
import { toolStart } from '../../shared/web-llm/gemma/markers.js'
import { textEnd } from '../../shared/web-llm/stream/markers.js'
import { QwenTags } from '../../shared/web-llm/qwen35/tags.js'

const groups = [
  ['<|channel>', toolStart],
  ['<channel|>'],
  [toolStart],
  ['<think>', '<tool_call>'],
  ['</think>'],
  ['</tool_call>'],
  QwenTags.markers,
]

it.each(groups.map((markers) => ({ markers })))('buffers $markers across chunks', ({ markers }) => {
  for (const marker of markers) {
    expect(marker.startsWith('<') && marker.lastIndexOf('<') === 0).toBe(true)
    for (let split = 1; split < marker.length; split++) {
      const partial = marker.slice(0, split)
      const first = '서울 😀 <literal> ' + partial
      const end = textEnd(first, markers)
      expect(first.slice(0, end)).toBe('서울 😀 <literal> ')
      expect(first.slice(end) + marker.slice(split)).toBe(marker)
    }
  }
})

it.each(groups.map((markers) => ({ markers })))(
  'releases literal text with $markers without buffering it',
  ({ markers }) => {
    for (const text of [
      '',
      '서울 😀',
      '<literal> plain text',
      '<<not a marker',
      'x'.repeat(200_000),
    ])
      expect(textEnd(text, markers)).toBe(text.length)
  },
)

it('keeps a shorter complete XML boundary until it can be scanned with the next chunk', () => {
  const text = 'text</function>'
  const end = textEnd(text, QwenTags.markers)
  expect(text.slice(0, end)).toBe('text')
  expect(text.slice(end)).toBe('</function>')
})
