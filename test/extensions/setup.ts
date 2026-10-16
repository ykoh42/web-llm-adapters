import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync } from 'node:fs'
import { resolve } from 'node:path'

// Official implementation linked by WebLLM. Pin the reviewed source used by these tests.
const revision = '33567d8bdba53184e8f5f24525f016d37d38df52'
const directory = resolve('.cache/cos-extension')
if (!existsSync(directory)) {
  mkdirSync('.cache', { recursive: true })
  execFileSync(
    'git',
    [
      'clone',
      '--no-checkout',
      'https://github.com/web-ai-community/cross-origin-storage-extension.git',
      directory,
    ],
    { stdio: 'inherit' },
  )
  execFileSync('git', ['-C', directory, 'checkout', '--detach', revision], { stdio: 'inherit' })
}
const current = execFileSync('git', ['-C', directory, 'rev-parse', 'HEAD'], {
  encoding: 'utf8',
}).trim()
if (current !== revision)
  throw new Error(`Expected Cross-Origin Storage revision ${revision}, got ${current}.`)
