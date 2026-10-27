import { readFileSync } from 'node:fs'
import { describe, expect, test } from 'vitest'
import { checkReport, type ModelReport } from '../scripts/release/report.js'
import { supportClaims } from '../scripts/release/support.js'

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
  test('accepts matching source and complete successful checks', () => {
    expect(checkReport(report, source, webllm, ['strands/model/tools'])).toEqual(report)
  })

  test('rejects stale code and WebLLM versions', () => {
    expect(() => checkReport(report, 'c'.repeat(64), webllm, [])).toThrow('stale')
    expect(() => checkReport(report, source, '0.2.87', [])).toThrow('stale')
  })

  test('does not count another adapter or capability as coverage', () => {
    expect(() => checkReport(report, source, webllm, ['pi/model/tools'])).toThrow('Missing')
    expect(() => checkReport(report, source, webllm, ['strands/model/thinking'])).toThrow('Missing')
  })

  test('requires cross-adapter checks to pass without replacing adapter coverage', () => {
    const integration = {
      adapter: 'integration' as const,
      name: 'custom model IDs',
      status: 'passed' as const,
      checks: ['model/tools'],
    }
    expect(
      checkReport({ ...report, tests: [...report.tests, integration] }, source, webllm, [
        'strands/model/tools',
      ]),
    ).toBeDefined()
    expect(() =>
      checkReport({ ...report, tests: [integration] }, source, webllm, ['strands/model/tools']),
    ).toThrow('Missing')
    expect(() =>
      checkReport(
        { ...report, tests: [...report.tests, { ...integration, status: 'failed' }] },
        source,
        webllm,
        [],
      ),
    ).toThrow('must pass')
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
})
