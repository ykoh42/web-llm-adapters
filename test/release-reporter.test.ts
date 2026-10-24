import { afterEach, expect, test, vi } from 'vitest'
import type { FullResult, TestCase, TestResult } from '@playwright/test/reporter'
import ModelReporter from '../scripts/release/reporter.js'

const result: FullResult = { status: 'passed', startTime: new Date(), duration: 0 }

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllEnvs()
})

test('report initialization errors fail the run instead of being swallowed by Playwright', async () => {
  vi.stubEnv('WEB_LLM_RELEASE_ENVIRONMENT', '')
  const log = vi.spyOn(console, 'error').mockImplementation(() => {})
  const reporter = new ModelReporter()
  reporter.onBegin()
  expect(await reporter.onEnd(result)).toEqual({ status: 'failed' })
  expect(log).toHaveBeenCalledOnce()
})

test('test collection errors fail the run even if every test itself passed', async () => {
  vi.stubEnv(
    'WEB_LLM_RELEASE_ENVIRONMENT',
    JSON.stringify({
      platform: 'darwin',
      arch: 'arm64',
      node: 'v24.11.0',
      browser: '156',
      gpu: { vendor: 'apple', architecture: 'common-3', device: '', description: '' },
    }),
  )
  const log = vi.spyOn(console, 'error').mockImplementation(() => {})
  const reporter = new ModelReporter()
  reporter.onBegin()
  reporter.onTestEnd(
    { parent: { project: () => ({ name: 'unknown' }) } } as unknown as TestCase,
    { status: 'passed' } as TestResult,
  )
  expect(await reporter.onEnd(result)).toEqual({ status: 'failed' })
  expect(log).toHaveBeenCalledOnce()
})

test('listing tests does not create a verification report', async () => {
  expect(await new ModelReporter().onEnd(result)).toBeUndefined()
})
