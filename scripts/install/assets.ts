import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { lstat, readFile } from 'node:fs/promises'

const [name, manager] = process.argv.slice(2)
assert.ok(name)
assert.equal(Boolean(process.versions.pnp), manager === 'yarn-pnp')
if (manager === 'yarn-pnp') {
  await assert.rejects(lstat('node_modules'), { code: 'ENOENT' })
} else {
  const packageDirectory = await lstat(`node_modules/${name}`)
  if (manager === 'pnpm') assert.ok(packageDirectory.isSymbolicLink())
  else assert.ok(packageDirectory.isDirectory())
}

const wasm = new URL(import.meta.resolve(`${name}/runtime/gemma-4-e2b.wasm`))
const bytes = await readFile(wasm)
assert.ok(WebAssembly.validate(bytes))
const manifest = JSON.parse(await readFile(new URL('manifest.json', wasm), 'utf8'))
assert.equal('sha256-' + createHash('sha256').update(bytes).digest('base64'), manifest.integrity)
for (const [file, hash] of Object.entries(manifest.files)) {
  const data = await readFile(new URL(file, wasm))
  assert.equal(createHash('sha256').update(data).digest('hex'), hash, file)
}
