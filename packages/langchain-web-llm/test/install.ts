import { runtimeFiles } from '../../../shared/build.ts'
import { execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../', import.meta.url))
const temporary = await mkdtemp(join(tmpdir(), 'langchain-web-llm-'))
const run = (cmd: string, args: string[], cwd = temporary) =>
  execFileSync(cmd, args, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] })
try {
  const [pack] = JSON.parse(
    run('npm', ['pack', '--ignore-scripts', '--json', '--pack-destination', temporary], root),
  )
  const wasm = pack.files.find(
    ({ path }: { path: string }) => path === 'dist/runtime/gemma-4-e2b.wasm',
  )
  if (!wasm || (wasm.mode & 0o111) !== 0)
    throw new Error('Wasm package file must not be executable.')
  const allowed = new Set([
    'README.md',
    'LICENSE',
    'CHANGELOG.md',
    'package.json',
    'dist/index.js',
    'dist/index.d.ts',
    ...runtimeFiles.map((name) => 'dist/runtime/' + name),
  ])
  const packed = new Set<string>(pack.files.map(({ path }: { path: string }) => path))
  for (const path of allowed) {
    if (!packed.has(path)) throw new Error(`Missing npm file: ${path}`)
  }
  for (const { path } of pack.files) {
    if (!allowed.has(path)) throw new Error(`Unexpected npm file: ${path}`)
  }
  await writeFile(
    join(temporary, 'package.json'),
    JSON.stringify({ private: true, type: 'module' }),
  )
  run('npm', [
    'install',
    '--ignore-scripts',
    '--no-fund',
    '--no-audit',
    join(temporary, pack.filename),
    '@langchain/core@1.2.18',
    '@mlc-ai/web-llm@0.2.86',
    'typescript@6.0.3',
    '@types/node@24.19.2',
  ])
  const audit = JSON.parse(run('npm', ['audit', '--omit=dev', '--json']))
  if (audit.metadata.vulnerabilities.total)
    throw new Error('Published dependencies failed npm audit.')
  const packageRoot = join(temporary, 'node_modules/langchain-web-llm')
  for (const { path } of pack.files) {
    const bytes = await readFile(join(packageRoot, path))
    for (const prefix of [
      '/Users/',
      '/home/',
      'npm_' + 'token=',
      '-----BEGIN ' + 'PRIVATE KEY-----',
    ])
      if (bytes.includes(Buffer.from(prefix))) throw new Error(`Private data in npm file: ${path}`)
  }
  const source = await readFile(join(packageRoot, 'dist/index.js'), 'utf8')
  if (source.includes('@strands-agents/'))
    throw new Error('The LangChain package depends on Strands.')
  const runtime = join(packageRoot, 'dist/runtime')
  const manifest = JSON.parse(await readFile(join(runtime, 'manifest.json'), 'utf8'))
  for (const [name, hash] of Object.entries(manifest.files)) {
    const bytes = await readFile(join(runtime, name))
    if (createHash('sha256').update(bytes).digest('hex') !== hash)
      throw new Error('Runtime hash mismatch: ' + name)
  }
  if (!WebAssembly.validate(await readFile(join(runtime, 'gemma-4-e2b.wasm'))))
    throw new Error('Invalid Wasm module.')
  await writeFile(
    join(temporary, 'consumer.ts'),
    `
import assert from 'node:assert/strict'
import { MLCEngine } from '@mlc-ai/web-llm'
import { ChatWebLLM, BrowserChatWebLLM, WebLLMEmbeddings, createBrowserBridge, withEngine } from 'langchain-web-llm'
const engine = new MLCEngine({logLevel:'SILENT'})
engine.chat.completions.create = (async () => (async function*(){yield {id:'test',object:'chat.completion.chunk',created:0,model:'test',choices:[{index:0,delta:{content:'Hello'},finish_reason:'stop'}]}})()) as unknown as typeof engine.chat.completions.create
const model = new ChatWebLLM({engine,modelId:'test'})
assert.equal((await model.invoke('Hi')).text,'Hello')
assert.equal((await model.batch(['One','Two'])).length,2)
assert.equal(typeof model.bindTools([]).invoke,'function')
assert.equal(typeof model.withStructuredOutput({type:'object'}).invoke,'function')
assert.equal(typeof new WebLLMEmbeddings({engine,modelId:'embedding'}).embedQuery,'function')
const bridge = createBrowserBridge(engine)
assert.equal(typeof bridge.start,'function')
const proxy = new BrowserChatWebLLM({evaluate:async () => {throw new Error('transport')}},{modelId:'test'})
assert.equal(proxy.modelId,'test')
await withEngine(engine, async native => assert.equal(native,engine))
await bridge.close()
`,
  )
  for (const [module, resolution] of [
    ['ESNext', 'Bundler'],
    ['NodeNext', 'NodeNext'],
  ])
    run('node', [
      'node_modules/typescript/bin/tsc',
      '--noEmit',
      '--strict',
      '--skipLibCheck',
      '--types',
      'node',
      '--target',
      'ES2022',
      '--module',
      module!,
      '--moduleResolution',
      resolution!,
      'consumer.ts',
    ])
  run('node', ['consumer.ts'])
  console.log(
    `langchain-web-llm: installed tarball, public types, runtime hashes, runtime smoke, and dependency audit passed (${pack.files.length} files).`,
  )
} finally {
  await rm(temporary, { recursive: true, force: true })
}
