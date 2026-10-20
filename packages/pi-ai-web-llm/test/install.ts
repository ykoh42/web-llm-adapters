import { runtimeFiles } from '../../../shared/build.ts'
import { execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../', import.meta.url))
const temporary = await mkdtemp(join(tmpdir(), 'pi-ai-web-llm-'))
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
    '@earendil-works/pi-ai@1.3.0',
    '@earendil-works/pi-agent-core@1.3.0',
    '@mlc-ai/web-llm@0.2.86',
    'typescript@6.0.3',
    '@types/node@24.19.2',
  ])
  const audit = JSON.parse(run('npm', ['audit', '--omit=dev', '--json']))
  if (audit.metadata.vulnerabilities.total)
    throw new Error('Published dependencies failed npm audit.')
  const packageRoot = join(temporary, 'node_modules/pi-ai-web-llm')
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
  if (source.includes('@strands-agents/') || source.includes('@langchain/'))
    throw new Error('The Pi AI adapter depends on another framework.')
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
import { createModels } from '@earendil-works/pi-ai/models'
import { Agent } from '@earendil-works/pi-agent-core'
import { createWebLLMProvider } from 'pi-ai-web-llm'
const engine = new MLCEngine({logLevel:'SILENT'})
engine.chat.completions.create = (async () => (async function*(){yield {id:'test',object:'chat.completion.chunk',created:0,model:'test',choices:[{index:0,delta:{content:'Hello'},finish_reason:'stop'}]}})()) as unknown as typeof engine.chat.completions.create
const provider = createWebLLMProvider({engine,modelId:'test',contextWindow:8192})
const models = createModels()
models.setProvider(provider)
const model = provider.getModels()[0]!
const api: 'webllm' = model.api
assert.equal(api,'webllm')
const stream = models.stream(model,{messages:[{role:'user',content:'Hi',timestamp:0}]},{request:{seed:7}})
const events: string[] = []
for await (const event of stream) events.push(event.type)
assert.deepEqual(events,['start','text_start','text_delta','text_end','done'])
const response = await stream.result()
assert.equal(response.stopReason,'stop')
assert.deepEqual(response.content,[{type:'text',text:'Hello'}])
const agent = new Agent({initialState:{model},streamFn:models.streamSimple.bind(models)})
await agent.prompt('Hello')
assert.equal(agent.state.messages.at(-1)?.role,'assistant')

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
      '--exactOptionalPropertyTypes',
      '--noUncheckedIndexedAccess',
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
    `pi-ai-web-llm: installed tarball, public types, runtime hashes, runtime smoke, and dependency audit passed (${pack.files.length} files).`,
  )
} finally {
  await rm(temporary, { recursive: true, force: true })
}
