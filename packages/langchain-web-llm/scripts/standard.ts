import { createHash } from 'node:crypto'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { dirname } from 'node:path'
import source from '../test/standard/source.json' with { type: 'json' }

// The official suite is private on npm. Fetch its unchanged, pinned sources for Vitest.
const root = new URL('../../../.cache/langchain-standard/', import.meta.url)
await Promise.all(
  Object.entries(source.files).map(async ([name, hash]) => {
    const path = fileURLToPath(new URL(name, root))
    let bytes: Uint8Array
    try {
      bytes = await readFile(path)
    } catch {
      bytes = new Uint8Array()
    }
    if (createHash('sha256').update(bytes).digest('hex') === hash) return
    const url = `https://raw.githubusercontent.com/${source.repository}/${source.revision}/internal/standard-tests/src/${name}`
    const response = await fetch(url)
    if (!response.ok) throw new Error(`Could not fetch standard tests: ${response.status}`)
    bytes = new Uint8Array(await response.arrayBuffer())
    if (createHash('sha256').update(bytes).digest('hex') !== hash)
      throw new Error(`Standard test hash mismatch: ${name}`)
    await mkdir(dirname(path), { recursive: true })
    await writeFile(path, bytes)
  }),
)
