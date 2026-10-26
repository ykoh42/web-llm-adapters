import { chromium, type Browser, type FullConfig } from '@playwright/test'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'

const state = vi.hoisted(() => ({
  checkClean: vi.fn(),
}))

vi.mock('../scripts/release/source.ts', () => state)

const { default: setup } = await import('../scripts/release/setup.ts')
const previousEnvironment = process.env.WEB_LLM_RELEASE_ENVIRONMENT

beforeEach(() => {
  state.checkClean.mockReset()
})

afterEach(() => {
  vi.restoreAllMocks()
  if (previousEnvironment === undefined) delete process.env.WEB_LLM_RELEASE_ENVIRONMENT
  else process.env.WEB_LLM_RELEASE_ENVIRONMENT = previousEnvironment
})

const config = { projects: [{ retries: 0 }] } as FullConfig

test('rejects uncommitted source changes', async () => {
  state.checkClean.mockImplementation(() => {
    throw new Error('Commit source changes before verifying real models.')
  })

  await expect(setup(config)).rejects.toThrow('Commit source changes')
})

test('rejects a sharded run', async () => {
  await expect(setup({ ...config, shard: { current: 1, total: 2 } })).rejects.toThrow(
    'without retries',
  )
})

test('records the GPU environment and closes the browser', async () => {
  const page = {
    goto: vi.fn(),
    evaluate: vi.fn().mockResolvedValue({
      vendor: 'Apple',
      architecture: 'Apple GPU',
      device: '',
      description: '',
    }),
  }
  const browser = {
    newPage: vi.fn().mockResolvedValue(page),
    close: vi.fn(),
    version: vi.fn(() => '156'),
  } as unknown as Browser
  vi.spyOn(chromium, 'launch').mockResolvedValue(browser)

  await setup(config)

  expect(JSON.parse(process.env.WEB_LLM_RELEASE_ENVIRONMENT!)).toMatchObject({
    browser: '156',
    gpu: { vendor: 'Apple', architecture: 'Apple GPU' },
  })
  expect(browser.close).toHaveBeenCalledOnce()
})
