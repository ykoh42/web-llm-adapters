import { readFileSync } from 'node:fs'
import { describe, expect, test } from 'vitest'
import { prebuiltAppConfig } from '@mlc-ai/web-llm'
import { checkReport, checkTests, type ModelReport } from '../scripts/release/report.js'
import { recordThinking, supportClaims } from '../scripts/release/support.js'
import { hasToolResult } from './support/native-output.js'

const source = 'a'.repeat(64)
const webllm = '0.2.86'
const report: ModelReport = {
  commit: 'b'.repeat(40),
  source,
  webllm,
  checkedAt: '2026-10-24T00:00:00.000Z',
  environment: {
    platform: 'darwin',
    arch: 'arm64',
    node: 'v24.11.0',
    browser: '156',
    gpu: { vendor: 'apple', architecture: 'common-3', device: '', description: '' },
  },
  tests: [{ adapter: 'strands', name: 'tool loop', status: 'passed', checks: ['model/tools'] }],
}

describe('release model gate', () => {
  test('checks tool results in later native chat and completion requests', () => {
    const chat = { model: 'test', messages: [{ role: 'user' as const, content: 'Hello' }] }
    const result = { role: 'user' as const, content: '<tool_response>42</tool_response>' }
    expect(hasToolResult([chat], '42')).toBe(false)
    expect(hasToolResult([{ ...chat, messages: [{ role: 'user', content: '42' }] }], '42')).toBe(
      false,
    )
    expect(
      hasToolResult([chat, { ...chat, messages: [{ role: 'assistant', content: '42' }] }], '42'),
    ).toBe(false)
    expect(hasToolResult([chat, { ...chat, messages: [...chat.messages, result] }], '42')).toBe(
      true,
    )
    expect(
      hasToolResult([{ prompt: 'Hello' }, { prompt: '<|tool_response>42<tool_response|>' }], '42'),
    ).toBe(true)
    expect(hasToolResult([{ prompt: '42<|tool_response>other<tool_response|>' }], '42')).toBe(false)
    expect(hasToolResult([chat, chat], '42')).toBe(false)
  })

  test('accepts matching source and complete successful checks', () => {
    expect(checkReport(report, source, webllm, ['strands/model/tools'])).toEqual(report)
  })

  test('requires release tests even when support claims are already covered', () => {
    const expected = ['strands/tool loop', 'strands/protocol regression']
    expect(checkReport(report, source, webllm, ['strands/model/tools'])).toEqual(report)
    expect(() => checkTests(report, expected)).toThrow('strands/protocol regression')
    const tests = [
      ...report.tests,
      { ...report.tests[0]!, name: 'protocol regression', checks: [] },
    ]
    expect(() => checkTests({ ...report, tests }, expected)).not.toThrow()
    expect(() => checkTests({ ...report, tests: [...tests, tests[0]!] }, expected)).toThrow()
  })

  test('rejects stale code and WebLLM versions', () => {
    expect(() => checkReport(report, 'c'.repeat(64), webllm, [])).toThrow('stale')
    expect(() => checkReport(report, source, '0.2.87', [])).toThrow('stale')
  })

  test('does not count another adapter or capability as coverage', () => {
    expect(() => checkReport(report, source, webllm, ['pi/model/tools'])).toThrow('Missing')
    expect(() => checkReport(report, source, webllm, ['strands/model/thinking'])).toThrow('Missing')
  })

  test('requires observed reasoning in the next request before counting thinking support', () => {
    const info: { annotations: { type: string; description?: string }[] } = { annotations: [] }
    const model = 'Qwen3-4B'
    const claim = `strands/${model}/thinking`
    const first = { model, stream: true as const, messages: [] }
    const next = {
      ...first,
      messages: [{ role: 'assistant' as const, content: '<think>\nA thought.\n</think>' }],
    }
    recordThinking(info, model, ['', ' A final thought.'], [first, next])
    expect(() => checkReport(report, source, webllm, [claim])).toThrow(claim)
    expect(() => recordThinking(info, model, ['A thought.'], [first, first])).toThrow(
      'lost the native thinking block',
    )
    recordThinking(info, model, ['\nA thought.\n', ''], [first, next])
    const tests = [
      { ...report.tests[0]!, checks: info.annotations.map((item) => item.description) },
    ]
    expect(checkReport({ ...report, tests }, source, webllm, [claim])).toMatchObject({ tests })
  })

  test('checks Qwen3.5 and Gemma thinking in their native history formats', () => {
    const info: { annotations: { type: string; description?: string }[] } = { annotations: [] }
    const qwen = { model: 'Qwen3.5-4B', stream: true as const, messages: [] }
    const qwenNext = {
      ...qwen,
      messages: [{ role: 'assistant' as const, content: '<think>\nPlan.\n</think>' }],
    }
    recordThinking(info, qwen.model, [' \nPlan.\n '], [qwen, qwenNext])
    const gemma = { model: 'gemma-4-E2B', stream: true as const, prompt: '' }
    const gemmaNext = { ...gemma, prompt: '<|channel>thought\nPlan.\n<channel|>' }
    recordThinking(info, gemma.model, ['Plan.'], [gemma, gemmaNext])
    expect(info.annotations.map((item) => item.description)).toEqual([
      'Qwen3.5-4B/thinking',
      'gemma-4-E2B/thinking',
    ])
  })

  test('does not mistake an earlier identical thought for the current one', () => {
    const info: { annotations: { type: string; description?: string }[] } = { annotations: [] }
    const qwen = {
      model: 'Qwen3-4B',
      stream: true as const,
      messages: [{ role: 'assistant' as const, content: '<think>\nPlan.\n</think>' }],
    }
    expect(() => recordThinking(info, qwen.model, ['Plan.'], [qwen, qwen])).toThrow(
      'lost the native thinking block',
    )
    const qwenNext = { ...qwen, messages: [...qwen.messages, ...qwen.messages] }
    recordThinking(info, qwen.model, ['Plan.'], [qwen, qwenNext])

    const gemma = {
      model: 'gemma-4-E2B',
      stream: true as const,
      prompt: '<|channel>thought\nPlan.\n<channel|>',
    }
    expect(() => recordThinking(info, gemma.model, ['Plan.'], [gemma, gemma])).toThrow(
      'lost the native thinking block',
    )
    recordThinking(
      info,
      gemma.model,
      ['Plan.'],
      [gemma, { ...gemma, prompt: gemma.prompt.repeat(2) }],
    )
    expect(info.annotations).toHaveLength(2)
  })

  test('does not count default tools or protocol errors as a constrained tool loop', () => {
    const constrained = 'strands/model/constrained-tools'
    expect(() => checkReport(report, source, webllm, [constrained])).toThrow('Missing')
    const tests = [{ ...report.tests[0]!, checks: ['model/tool-protocol'] }]
    expect(() => checkReport({ ...report, tests }, source, webllm, [constrained])).toThrow(
      'Missing',
    )
    tests[0]!.checks.push('model/constrained-tools')
    expect(checkReport({ ...report, tests }, source, webllm, [constrained])).toMatchObject({
      tests,
    })
    expect(() =>
      checkReport({ ...report, tests }, source, webllm, ['strands/model/tools']),
    ).toThrow('Missing')
  })

  test.each(['failed', 'timedOut', 'skipped', 'interrupted'] as const)(
    'rejects %s tests even when a later attempt passes',
    (status) => {
      const tests = [{ ...report.tests[0]!, status }, ...report.tests]
      expect(() => checkReport({ ...report, tests }, source, webllm, [])).toThrow('must pass')
    },
  )

  test('rejects missing and malformed evidence', () => {
    for (const value of [null, {}, { ...report, tests: [] }, { ...report, commit: '../main' }])
      expect(() => checkReport(value, source, webllm, [])).toThrow()
  })
})

describe('README support coverage', () => {
  const row = '| `model` | 80 | 20 | O | O | X | O (LangChain, Strands) | X | All |'
  const table = '## Models\n' + row

  test('expands All and respects per-feature adapter scope', () => {
    const claims = supportClaims(table)
    for (const adapter of ['langchain', 'pi', 'strands']) {
      expect(claims).toContain(`${adapter}/model/tools`)
      expect(claims).toContain(`${adapter}/model/thinking`)
      expect(claims).toContain(`${adapter}/engine/cancel`)
      expect(claims).toContain(`${adapter}/engine/concurrency`)
      expect(claims).toContain(`${adapter}/engine/resume`)
    }
    expect(claims).toContain('langchain/model/audio')
    expect(claims).toContain('strands/model/audio')
    expect(claims).not.toContain('pi/model/audio')
    expect(claims.some((claim) => claim.endsWith('/image'))).toBe(false)
  })

  test('covers image and embedding model rows', () => {
    const claims = supportClaims(
      '## Models\n| `vision` | Image input | All |\n| `embed` | Embeddings, 384 dimensions | LangChain, Strands |',
    )
    expect(claims).toContain('pi/vision/image')
    expect(claims).toContain('langchain/embed/embeddings')
    expect(claims).not.toContain('pi/embed/embeddings')
  })

  test('requires text evidence for verified adapters without other verified features', () => {
    const textOnly = '| `text-only` | 80 | 20 | X | X | X | X | X | All |'
    const partial = '| `partial` | 80 | 20 | O (LangChain, Strands) | X | X | X | X | All |'
    const claims = supportClaims(`${table}\n${textOnly}\n${partial}`)
    for (const adapter of ['langchain', 'pi', 'strands'])
      expect(claims).toContain(`${adapter}/text-only/text`)
    expect(claims).toContain('pi/partial/text')
    expect(claims).not.toContain('langchain/partial/text')
    expect(claims).not.toContain('strands/partial/text')
    expect(() => checkReport(report, source, webllm, ['pi/partial/text'])).toThrow('Missing')
    const tests = [
      ...report.tests,
      { adapter: 'pi', name: 'chat', status: 'passed', checks: ['partial/text'] },
    ]
    expect(checkReport({ ...report, tests }, source, webllm, ['pi/partial/text'])).toMatchObject({
      tests,
    })
  })

  test('requires constrained loops and default protocol checks for conditional tools', () => {
    const claims = supportClaims(table.replace('| O |', '| O* |'))
    for (const adapter of ['langchain', 'pi', 'strands']) {
      expect(claims).toContain(`${adapter}/model/constrained-tools`)
      expect(claims).toContain(`${adapter}/model/tool-protocol`)
      expect(claims).not.toContain(`${adapter}/model/tools`)
    }
    const scoped = supportClaims(table.replace('| O |', '| O* (Strands) |'))
    expect(scoped).toContain('strands/model/constrained-tools')
    expect(scoped).not.toContain('pi/model/constrained-tools')
    expect(scoped).not.toContain('langchain/model/constrained-tools')
  })

  test('ignores package and development tables', () => {
    expect(
      supportClaims(
        '| `package` | guide | examples | npm |\n' +
          table +
          '\n## Development\n| `other` | value |',
      ),
    ).toEqual(supportClaims(table))
  })

  test.each(['new-model', '[new-model](https://example.com)'])(
    'rejects a support row whose model ID is not in backticks: %s',
    (model) => {
      const extra = row.replace('`model`', model)
      expect(() => supportClaims(`${table}\n${extra}`)).toThrow('model')
      expect(() => supportClaims(`${table}\n${extra.slice(2)}`)).toThrow('model')
    },
  )

  test('reads model rows with optional outer pipes', () => {
    const extra = row.replace('`model`', '`other`').slice(2, -2)
    expect(supportClaims(`${table}\n${extra}`)).toContain('pi/other/tools')
    const header = 'Model | Score | Speed | Tools | Think | Image | Audio | Video | Adapters'
    const divider = '--- | --- | --- | --- | --- | --- | --- | --- | ---'
    expect(supportClaims(`## Models\n${header}\n${divider}\n${extra}`)).toContain('pi/other/tools')
  })

  test.each([
    '',
    '## Models\n',
    table + '\n' + row,
    table.replace('All', 'Other'),
    table.replace('| O |', '| Maybe |'),
    table.replace('| O | O |', '| O | O* |'),
    '## Models\n| `model` | Unknown input | All |',
    table.replace(' | All |', ' | Pi |'),
  ])('rejects missing, ambiguous or unknown model claims: %s', (markdown) => {
    expect(() => supportClaims(markdown)).toThrow()
  })

  test('reads the current support table without claiming Pi audio', () => {
    const claims = supportClaims(readFileSync('README.md', 'utf8'))
    expect(claims).toContain('strands/gemma-4-E2B-it-q4f16_1-MLC/audio')
    expect(claims).not.toContain('pi/gemma-4-E2B-it-q4f16_1-MLC/audio')
  })

  test('lists model IDs from the installed WebLLM catalog', () => {
    const claims = supportClaims(readFileSync('README.md', 'utf8'))
    const models = new Set(prebuiltAppConfig.model_list.map((model) => model.model_id))
    const unknown = [...new Set(claims.map((claim) => claim.split('/')[1]))].filter(
      (model) => model !== 'engine' && !models.has(model!),
    )
    expect(unknown).toEqual([])
  })
})
