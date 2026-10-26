import { execFile } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'
import { afterEach, expect, test, vi } from 'vitest'
import type { FullResult, TestCase, TestResult } from '@playwright/test/reporter'
import ModelReporter from '../scripts/release/reporter.js'
import * as reports from '../scripts/release/report.js'
import * as source from '../scripts/release/source.js'

const removeReport = vi.hoisted(() => vi.fn())
vi.mock('node:fs', async (original) => ({
  ...(await original<typeof import('node:fs')>()),
  rmSync: removeReport,
}))

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
  vi.restoreAllMocks()
  vi.unstubAllEnvs()
})

test.each(['before', 'during'])(
  'invalidates the previous report on failure %s a run',
  async (when) => {
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const reporter = new ModelReporter()
    if (when === 'during') {
      vi.stubEnv('WEB_LLM_RELEASE_ENVIRONMENT', environment)
      reporter.onTestBegin()
    }
    expect(await reporter.onEnd({ ...result, status: 'failed' })).toEqual({ status: 'failed' })
    expect(removeReport).toHaveBeenCalledWith(source.reportPath, { force: true })
  },
)

test('reports failure even when the previous report cannot be removed', async () => {
  removeReport.mockImplementationOnce(() => {
    throw new Error('Cannot remove report')
  })
  const log = vi.spyOn(console, 'error').mockImplementation(() => {})

  expect(await new ModelReporter().onEnd({ ...result, status: 'failed' })).toEqual({
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

test('listing tests needs no GPU environment and leaves the verification report intact', async () => {
  const root = fileURLToPath(new URL('../', import.meta.url))
  const report = join(root, '.github/real-models.json')
  const previous = readFileSync(report, 'utf8')
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
      timeout: 10_000,
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
  expect(readFileSync(report, 'utf8')).toBe(previous)
}, 15_000)
