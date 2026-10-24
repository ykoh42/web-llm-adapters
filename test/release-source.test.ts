import { execFileSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, expect, test } from 'vitest'

const moduleUrl = new URL('../scripts/release/source.ts', import.meta.url).href
let directory: string

function git(...args: string[]) {
  execFileSync('git', args, { cwd: directory, stdio: 'pipe' })
}

function commit() {
  git('add', '.')
  git('-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-qm', 'Test source')
}

function source() {
  return execFileSync(
    process.execPath,
    [
      '--input-type=module',
      '-e',
      `import { checkClean, sourceHash } from ${JSON.stringify(moduleUrl)}; checkClean(); console.log(sourceHash())`,
    ],
    { cwd: directory, encoding: 'utf8', stdio: 'pipe' },
  ).trim()
}

beforeEach(() => {
  directory = mkdtempSync(join(tmpdir(), 'release-source-'))
  git('init', '-q')
  writeFileSync(join(directory, 'model.ts'), 'export const tokens = 4096\n')
  writeFileSync(join(directory, 'README.md'), 'Usage\n')
  commit()
})

afterEach(() => rmSync(directory, { recursive: true, force: true }))

test('source edits invalidate results, including after commit', () => {
  const original = source()
  writeFileSync(join(directory, 'model.ts'), 'export const tokens = 8192\n')
  expect(() => source()).toThrow()
  commit()
  expect(source()).not.toBe(original)
})

test('lockfile changes invalidate results', () => {
  const original = source()
  writeFileSync(join(directory, 'pnpm-lock.yaml'), 'lockfileVersion: 9\n')
  commit()
  expect(source()).not.toBe(original)
})

test('documentation-only commits preserve tested source', () => {
  const original = source()
  writeFileSync(join(directory, 'README.md'), 'New usage\n')
  commit()
  expect(source()).toBe(original)
})

test('untracked code cannot be omitted from a verification run', () => {
  writeFileSync(join(directory, 'other.ts'), 'export {}\n')
  expect(() => source()).toThrow()
})

test('a generated report can be added without invalidating its own source', () => {
  const original = source()
  mkdirSync(join(directory, '.github'))
  writeFileSync(join(directory, '.github/real-models.json'), '{}\n')
  expect(source()).toBe(original)
  commit()
  expect(source()).toBe(original)
})
