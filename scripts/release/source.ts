import { execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'

export const reportPath = '.github/real-models.json'

export function webllmVersion() {
  return JSON.parse(readFileSync('node_modules/@mlc-ai/web-llm/package.json', 'utf8'))
    .version as string
}

export function git(...args: string[]) {
  return execFileSync('git', args, { encoding: 'utf8' }).trim()
}

export function checkClean() {
  const changes = git(
    'status',
    '--porcelain',
    '--untracked-files=normal',
    '--',
    '.',
    `:(exclude)${reportPath}`,
  )
  if (changes) throw new Error('Commit source changes before verifying real models.')
}

export function sourceHash() {
  // Documentation and the generated report can change without invalidating tested code.
  const entries = git('ls-tree', '-r', '-z', 'HEAD')
    .split('\0')
    .filter((entry) => {
      const path = entry.split('\t')[1]
      return path && path !== reportPath && !path.endsWith('.md')
    })
  return createHash('sha256').update(entries.join('\0')).digest('hex')
}
