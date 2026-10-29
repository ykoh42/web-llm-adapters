import { execFile, execFileSync } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'
import { afterEach, expect, test, vi } from 'vitest'
import type { FullResult, TestCase, TestResult } from '@playwright/test/reporter'
import * as oxfmt from 'oxfmt'
import ModelReporter from '../scripts/release/reporter.js'
import * as reports from '../scripts/release/report.js'
import * as source from '../scripts/release/source.js'
import * as suite from '../scripts/release/tests.js'

const removeReport = vi.hoisted(() => vi.fn())
const writeReport = vi.hoisted(() => vi.fn())
vi.mock('node:fs', async (original) => ({
  ...(await original<typeof import('node:fs')>()),
  rmSync: removeReport,
  writeFileSync: writeReport,
}))
vi.mock('oxfmt', async (original) => {
  const actual = await original<typeof import('oxfmt')>()
  return { ...actual, format: vi.fn(actual.format) }
})

const result: FullResult = { status: 'passed', startTime: new Date(), duration: 0 }
const run = promisify(execFile)
const environment = JSON.stringify({
  platform: 'darwin',
  arch: 'arm64',
  node: 'v24.11.0',
  browser: '156',
  gpu: { vendor: 'apple', architecture: 'common-3', device: '', description: '' },
})

afterEach(() => {
  removeReport.mockClear()
  writeReport.mockClear()
  vi.restoreAllMocks()
  vi.unstubAllEnvs()
})

test('keeps the previous report when setup fails before a run', async () => {
  expect(await new ModelReporter().onEnd({ ...result, status: 'failed' })).toEqual({
    status: 'failed',
  })
  expect(removeReport).not.toHaveBeenCalled()
})

test('invalidates the previous report when a run fails', async () => {
  vi.spyOn(console, 'error').mockImplementation(() => {})
  vi.stubEnv('WEB_LLM_RELEASE_ENVIRONMENT', environment)
  const reporter = new ModelReporter()
  reporter.onTestBegin()
  expect(await reporter.onEnd({ ...result, status: 'failed' })).toEqual({ status: 'failed' })
  expect(removeReport).toHaveBeenCalledWith(source.reportPath, { force: true })
})

test('reports failure even when the previous report cannot be removed', async () => {
  removeReport.mockImplementationOnce(() => {
    throw new Error('Cannot remove report')
  })
  const log = vi.spyOn(console, 'error').mockImplementation(() => {})
  vi.stubEnv('WEB_LLM_RELEASE_ENVIRONMENT', environment)
  const reporter = new ModelReporter()
  reporter.onTestBegin()

  expect(await reporter.onEnd({ ...result, status: 'failed' })).toEqual({
    status: 'failed',
  })
  expect(log).toHaveBeenCalledWith(expect.objectContaining({ message: 'Cannot remove report' }))
})

test('report initialization errors fail the run instead of being swallowed by Playwright', async () => {
  vi.stubEnv('WEB_LLM_RELEASE_ENVIRONMENT', '')
  const log = vi.spyOn(console, 'error').mockImplementation(() => {})
  const reporter = new ModelReporter()
  reporter.onTestBegin()
  expect(await reporter.onEnd(result)).toEqual({ status: 'failed' })
  expect(log).toHaveBeenCalledOnce()
  expect(removeReport).toHaveBeenCalledWith(source.reportPath, { force: true })
})

test('test collection errors fail the run even if every test itself passed', async () => {
  vi.stubEnv('WEB_LLM_RELEASE_ENVIRONMENT', environment)
  const log = vi.spyOn(console, 'error').mockImplementation(() => {})
  const reporter = new ModelReporter()
  reporter.onTestBegin()
  reporter.onTestEnd(
    { parent: { project: () => ({ name: 'unknown' }) } } as unknown as TestCase,
    { status: 'passed' } as TestResult,
  )
  expect(await reporter.onEnd(result)).toEqual({ status: 'failed' })
  expect(log).toHaveBeenCalledOnce()
})

test('keeps every result when Playwright starts another test', async () => {
  vi.stubEnv('WEB_LLM_RELEASE_ENVIRONMENT', environment)
  vi.spyOn(source, 'checkClean').mockImplementation(() => {})
  vi.spyOn(console, 'error').mockImplementation(() => {})
  const check = vi.spyOn(reports, 'checkReport').mockImplementation(() => {
    throw new Error('Report inspected without writing it.')
  })
  const reporter = new ModelReporter()
  for (const name of ['first', 'second']) {
    reporter.onTestBegin()
    reporter.onTestEnd(
      {
        parent: { project: () => ({ name: 'strands' }) },
        titlePath: () => ['', 'strands', name],
        annotations: [],
      } as unknown as TestCase,
      { status: 'passed' } as TestResult,
    )
  }
  await reporter.onEnd(result)
  expect(check.mock.calls[0]?.[0]).toMatchObject({
    tests: [
      { name: 'first', status: 'passed' },
      { name: 'second', status: 'passed' },
    ],
  })
})

test('rejects a partial run even when its support checks pass', async () => {
  vi.stubEnv('WEB_LLM_RELEASE_ENVIRONMENT', environment)
  vi.spyOn(source, 'checkClean').mockImplementation(() => {})
  vi.spyOn(console, 'error').mockImplementation(() => {})
  vi.spyOn(reports, 'checkReport').mockImplementation((report) => report as reports.ModelReport)
  vi.spyOn(suite, 'releaseTests').mockReturnValue(['strands/first', 'strands/second'])
  const reporter = new ModelReporter()
  reporter.onTestBegin()
  reporter.onTestEnd(
    {
      parent: { project: () => ({ name: 'strands' }) },
      titlePath: () => ['', 'strands', 'first'],
      annotations: [],
    } as unknown as TestCase,
    { status: 'passed' } as TestResult,
  )

  expect(await reporter.onEnd(result)).toEqual({ status: 'failed' })
  expect(removeReport).toHaveBeenCalledWith(source.reportPath, { force: true })
})

test('writes a complete report using the project format', async () => {
  vi.stubEnv('WEB_LLM_RELEASE_ENVIRONMENT', environment)
  vi.spyOn(source, 'checkClean').mockImplementation(() => {})
  vi.spyOn(reports, 'checkReport').mockImplementation((report) => report as reports.ModelReport)
  vi.spyOn(suite, 'releaseTests').mockReturnValue(['strands/first'])
  const reporter = new ModelReporter()
  reporter.onTestBegin()
  reporter.onTestEnd(
    {
      parent: { project: () => ({ name: 'strands' }) },
      titlePath: () => ['', 'strands', 'first'],
      annotations: [],
    } as unknown as TestCase,
    { status: 'passed' } as TestResult,
  )

  expect(await reporter.onEnd(result)).toEqual({ status: 'passed' })
  expect(writeReport).toHaveBeenCalledOnce()
  const [path, content] = writeReport.mock.calls[0] as [string, string]
  expect(path).toBe(source.reportPath)
  expect(JSON.parse(content)).toMatchObject({
    tests: [{ adapter: 'strands', name: 'first', status: 'passed', checks: [] }],
  })
  const root = fileURLToPath(new URL('../', import.meta.url))
  const formatted = execFileSync(
    process.execPath,
    [join(root, 'node_modules/oxfmt/bin/oxfmt'), '--stdin-filepath', path],
    { cwd: root, encoding: 'utf8', input: content },
  )
  expect(formatted).toBe(content)
})

test('fails the run if formatting the report fails', async () => {
  const invalid = await oxfmt.format(source.reportPath, '{ invalid }')
  expect(invalid.errors.length).toBeGreaterThan(0)
  vi.mocked(oxfmt.format).mockResolvedValueOnce(invalid)
  vi.spyOn(console, 'error').mockImplementation(() => {})
  vi.stubEnv('WEB_LLM_RELEASE_ENVIRONMENT', environment)
  vi.spyOn(source, 'checkClean').mockImplementation(() => {})
  vi.spyOn(reports, 'checkReport').mockImplementation((report) => report as reports.ModelReport)
  vi.spyOn(suite, 'releaseTests').mockReturnValue(['strands/first'])
  const reporter = new ModelReporter()
  reporter.onTestBegin()
  reporter.onTestEnd(
    {
      parent: { project: () => ({ name: 'strands' }) },
      titlePath: () => ['', 'strands', 'first'],
      annotations: [],
    } as unknown as TestCase,
    { status: 'passed' } as TestResult,
  )

  expect(await reporter.onEnd(result)).toEqual({ status: 'failed' })
  expect(writeReport).not.toHaveBeenCalled()
  expect(removeReport).toHaveBeenCalledWith(source.reportPath, { force: true })
})

test('normalizes file paths without changing test titles', () => {
  expect(suite.testName(['tests\\models.spec.ts', 'a\\b'])).toBe('tests/models.spec.ts > a\\b')
})

test('collects the full release suite even when model selection is set', () => {
  vi.stubEnv('MODEL_IDS', 'Qwen3.5-4B-q4f16_1-MLC')
  const tests = suite.releaseTests()
  expect(tests.some((name) => name.includes('Qwen3.5-9B-q4f16_1-MLC'))).toBe(true)
  expect(tests.some((name) => name.includes('Qwen2.5-0.5B-Instruct-q4f16_1-MLC'))).toBe(true)
}, 40_000)

test('listing tests needs no GPU environment and leaves the verification report intact', async () => {
  const root = fileURLToPath(new URL('../', import.meta.url))
  const report = join(root, '.github/real-models.json')
  const previous = existsSync(report) ? readFileSync(report, 'utf8') : undefined
  const { stdout, stderr } = await run(
    process.execPath,
    [
      join(root, 'node_modules/@playwright/test/cli.js'),
      'test',
      '--config',
      'playwright.release.config.ts',
      '--list',
    ],
    {
      cwd: root,
      timeout: 30_000,
      env: {
        ...process.env,
        WEB_LLM_RELEASE_ENVIRONMENT: undefined,
        FORCE_COLOR: undefined,
        NO_COLOR: undefined,
      },
    },
  )
  expect(stdout).toMatch(/Total: \d+ tests in \d+ files/)
  expect(stderr).toBe('')
  expect(existsSync(report) ? readFileSync(report, 'utf8') : undefined).toBe(previous)
}, 40_000)
