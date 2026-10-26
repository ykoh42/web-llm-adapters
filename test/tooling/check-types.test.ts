import { mkdtempSync, mkdirSync, readdirSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { checkTypes } from '../../scripts/check-types.ts'

let directory: string

function addPackage(name: string, declaration: string) {
  const root = join(directory, 'fixtures', name)
  mkdirSync(join(root, 'dist/runtime'), { recursive: true })
  writeFileSync(
    join(root, 'package.json'),
    JSON.stringify({
      name,
      type: 'module',
      exports: {
        '.': { types: './dist/index.d.ts' },
        './runtime/manifest.json': './dist/runtime/manifest.json',
      },
    }),
  )
  writeFileSync(join(root, 'dist/index.d.ts'), declaration)
  writeFileSync(join(root, 'dist/runtime/manifest.json'), '{}')
  symlinkSync(root, join(directory, 'node_modules', name), 'junction')
}

beforeEach(() => {
  directory = mkdtempSync(join(tmpdir(), 'consumer-types-'))
  mkdirSync(join(directory, 'node_modules/@types'), { recursive: true })
  writeFileSync(join(directory, 'package.json'), '{"type":"module"}')
  writeFileSync(
    join(directory, 'consumer.ts'),
    "import { answer } from 'public-types'; export const result: number = answer",
  )
  symlinkSync(
    dirname(fileURLToPath(import.meta.resolve('@types/node/package.json'))),
    join(directory, 'node_modules/@types/node'),
    'junction',
  )
})

afterEach(() => {
  try {
    expect(readdirSync(directory).some((name) => name.startsWith('.typecheck-'))).toBe(false)
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

// Allow cold compiler startup under coverage.
describe.each(['typescript', '@typescript/native'])(
  '%s consumer diagnostics',
  { timeout: 30_000 },
  (compiler) => {
    beforeEach(() => {
      symlinkSync(
        dirname(fileURLToPath(import.meta.resolve(compiler + '/package.json'))),
        join(directory, 'node_modules/typescript'),
        'junction',
      )
    })

    it('checks valid public declarations in both module modes', () => {
      addPackage('public-types', 'export declare const answer: number')
      expect(() => checkTypes(directory, 'public-types')).not.toThrow()
    })

    it('rejects an invalid public type even when the consumer does not use it', () => {
      addPackage(
        'public-types',
        'export declare const answer: number; export type Broken = MissingPublicType',
      )
      expect(() => checkTypes(directory, 'public-types')).toThrow('MissingPublicType')
    })

    it('rejects invalid consumer code', () => {
      addPackage('public-types', 'export declare const answer: string')
      expect(() => checkTypes(directory, 'public-types')).toThrow()
    })

    it('also reports errors that only appear with NodeNext resolution', () => {
      addPackage('public-types', 'export declare const answer: number')
      writeFileSync(join(directory, 'helper.ts'), 'export const extra = 1')
      writeFileSync(
        join(directory, 'consumer.ts'),
        "import { answer } from 'public-types'; import { extra } from './helper'; export const result = answer + extra",
      )
      expect(() => checkTypes(directory, 'public-types')).toThrow('explicit file extensions')
    })

    it('checks owned types without taking over dependency type checking', () => {
      addPackage('peer-types', 'export type Peer = number; export type Broken = MissingPeerType')
      addPackage(
        'public-types',
        "import type { Peer } from 'peer-types'; export declare const answer: Peer",
      )
      expect(() => checkTypes(directory, 'public-types')).not.toThrow()
    })
  },
)
