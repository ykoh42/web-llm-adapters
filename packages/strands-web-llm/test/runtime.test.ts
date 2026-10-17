import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { describe, expect, it } from 'vitest'
import manifest from '../runtime/manifest.json'

describe('packaged runtime', () => {
  it('contains no developer home paths', async () => {
    const bytes = await readFile(new URL('../runtime/gemma-4-e2b.wasm', import.meta.url))
    for (const prefix of ['/Users/', '/home/', '\\Users\\']) {
      expect(bytes.includes(Buffer.from(prefix))).toBe(false)
    }
  })

  it('matches its recorded hashes and WebLLM integrity value', async () => {
    for (const [name, hash] of Object.entries(manifest.files)) {
      const bytes = await readFile(new URL('../runtime/' + name, import.meta.url))
      expect(createHash('sha256').update(bytes).digest('hex')).toBe(hash)
    }
    const bytes = await readFile(new URL('../runtime/gemma-4-e2b.wasm', import.meta.url))
    expect('sha256-' + createHash('sha256').update(bytes).digest('base64')).toBe(manifest.integrity)
  })
})
