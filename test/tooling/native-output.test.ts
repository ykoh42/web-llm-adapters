import { expect, it } from 'vitest'
import type { Completion } from '@mlc-ai/web-llm'
import { chunk } from '../../fixtures/web-llm.js'
import { nativeOutput } from '../support/native-output.js'

it('keeps code in native thinking instead of copying it into the answer', () => {
  const native = [
    chunk({ content: '<think>code-999dca88</think>\nThe task is fully resolved.' }, 'stop'),
  ]
  expect(nativeOutput(native, 'Qwen3-1.7B-q4f16_1-MLC')).toEqual({
    text: '\nThe task is fully resolved.',
    thinking: ['code-999dca88'],
  })
})

it('restores a Qwen thinking prefix without trimming either block', () => {
  expect(
    nativeOutput(
      [chunk({ content: '\n Why?\n</think>\n Answer. ' }, 'stop')],
      'Qwen3.5-4B-q4f16_1-MLC',
      '<think>',
    ),
  ).toEqual({ text: '\n Answer. ', thinking: ['\n Why?\n'] })
})

it('reads prefilled Gemma thinking through raw completions', () => {
  const native: Completion = {
    id: 'test',
    object: 'text_completion',
    created: 0,
    model: 'gemma-4-E2B-it-q4f16_1-MLC',
    choices: [{ index: 0, text: 'Why?<channel|>Answer.', finish_reason: 'stop' }],
  }
  const first: Completion = {
    ...native,
    id: 'first',
    choices: [
      { index: 0, text: '<|channel>thought\nFirst.<channel|>Calling.', finish_reason: 'stop' },
    ],
  }
  expect(nativeOutput([first, native], native.model, ['', '<|channel>thought\n'])).toEqual({
    text: 'Answer.',
    thinking: ['First.', 'Why?'],
  })
})

it('keeps literal thinking tags for a model without thinking', () => {
  expect(
    nativeOutput([chunk({ content: '<think>literal</think>' }, 'stop')], 'Qwen2.5-3B-Instruct'),
  ).toEqual({ text: '<think>literal</think>', thinking: [''] })
})

it('ignores earlier responses and retains the last reply before its usage chunk', () => {
  const native = [
    {
      ...chunk({ content: '<think>Earlier thought.</think>Earlier reply.' }, 'stop'),
      id: 'earlier',
    },
    chunk({ content: 'Final ' }),
    chunk({ content: 'reply.' }, 'stop'),
    { ...chunk(), choices: [] },
  ]
  expect(nativeOutput(native, 'Qwen3-4B')).toEqual({
    text: 'Final reply.',
    thinking: ['Earlier thought.', ''],
  })
})

it('rejects missing native output', () => {
  expect(() => nativeOutput([], 'Qwen3-4B')).toThrow('No native response was recorded.')
})
