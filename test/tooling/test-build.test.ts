import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type { TestProject } from 'vitest/node'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import setup from '../../scripts/test-build.ts'

let directory: string
let source: string
let output: string

beforeEach(() => {
  directory = mkdtempSync(join(tmpdir(), 'adapter-build-'))
  source = join(directory, 'src/index.js')
  output = join(directory, 'packages/fixture/dist/index.js')
  mkdirSync(join(directory, 'src'))
  mkdirSync(join(directory, 'packages/fixture/dist'), { recursive: true })
  writeFileSync(source, 'export const value = "current"\n')
  writeFileSync(output, 'export const value = "stale"\n')
  writeFileSync(
    join(directory, 'package.json'),
    JSON.stringify({ type: 'module', scripts: { 'build:adapters': 'node build.mjs' } }),
  )
  writeFileSync(
    join(directory, 'build.mjs'),
    "import { copyFileSync } from 'node:fs'; copyFileSync('src/index.js', 'packages/fixture/dist/index.js')",
  )
})

afterEach(() => rmSync(directory, { recursive: true, force: true }))

function project(watch: boolean) {
  const provide = vi.fn()
  const invalidateFile = vi.fn()
  let rerun: (() => Promise<void>) | undefined
  const value = {
    config: { root: directory },
    provide,
    vitest: { config: { watch }, invalidateFile },
    onTestsRerun: (run: () => Promise<void>) => {
      rerun = run
    },
  } as unknown as TestProject
  return { value, provide, invalidateFile, rerun: () => rerun!() }
}

it('rebuilds package exports for a direct non-watch test run', async () => {
  const runner = project(false)

  await setup(runner.value)

  expect(readFileSync(output, 'utf8')).toBe(readFileSync(source, 'utf8'))
})

it('fails a non-watch run when package exports cannot be rebuilt', async () => {
  const runner = project(false)
  writeFileSync(join(directory, 'build.mjs'), 'throw new Error("Build failed")')

  await expect(setup(runner.value)).rejects.toThrow('Build failed')
})

it('rebuilds and invalidates package exports before a watch rerun', async () => {
  const runner = project(true)
  await setup(runner.value)
  writeFileSync(source, 'export const value = "changed"\n')

  await runner.rerun()

  expect(readFileSync(output, 'utf8')).toBe(readFileSync(source, 'utf8'))
  expect(runner.invalidateFile).toHaveBeenCalledWith(output.replaceAll('\\', '/'))
})

it('waits for the latest build when a second save happens during a watch rebuild', async () => {
  const runner = project(true)
  const count = join(directory, 'build-count')
  writeFileSync(
    join(directory, 'build.mjs'),
    [
      "import { existsSync, readFileSync, writeFileSync } from 'node:fs'",
      "import { setTimeout } from 'node:timers/promises'",
      "const count = Number(existsSync('build-count') ? readFileSync('build-count', 'utf8') : 0) + 1",
      "const source = readFileSync('src/index.js', 'utf8')",
      "writeFileSync('build-count', String(count))",
      "if (existsSync('hold-builds')) while (!existsSync(`continue-${count}`)) await setTimeout(10)",
      "writeFileSync('packages/fixture/dist/index.js', source)",
    ].join('\n'),
  )
  await setup(runner.value)
  writeFileSync(join(directory, 'hold-builds'), '')
  writeFileSync(source, 'export const value = "first"\n')

  const runs = [runner.rerun()]
  try {
    await vi.waitFor(() => expect(readFileSync(count, 'utf8')).toBe('2'), { timeout: 10000 })
    writeFileSync(source, 'export const value = "latest"\n')
    runs.push(runner.rerun())
    writeFileSync(join(directory, 'continue-2'), '')
    await vi.waitFor(() => expect(readFileSync(count, 'utf8')).toBe('3'), { timeout: 10000 })
    writeFileSync(join(directory, 'continue-3'), '')
    await Promise.all(runs)

    expect(readFileSync(output, 'utf8')).toBe('export const value = "latest"\n')
  } finally {
    writeFileSync(join(directory, 'continue-2'), '')
    writeFileSync(join(directory, 'continue-3'), '')
    await Promise.allSettled(runs)
  }
}, 30_000)

it('keeps watching after a build failure and clears the error after a successful build', async () => {
  const runner = project(true)
  writeFileSync(join(directory, 'build.mjs'), 'throw new Error("Build failed")')
  await setup(runner.value)
  expect(runner.provide).toHaveBeenLastCalledWith(
    'adapterBuildError',
    expect.stringContaining('Build failed'),
  )
  writeFileSync(
    join(directory, 'build.mjs'),
    "import { copyFileSync } from 'node:fs'; copyFileSync('src/index.js', 'packages/fixture/dist/index.js')",
  )

  await runner.rerun()

  expect(runner.provide).toHaveBeenLastCalledWith('adapterBuildError', undefined)
  expect(readFileSync(output, 'utf8')).toBe(readFileSync(source, 'utf8'))
})
