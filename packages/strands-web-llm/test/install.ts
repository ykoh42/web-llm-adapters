import { runtimeFiles } from '../../../shared/build.ts'
import { execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../', import.meta.url))
const temporary = await mkdtemp(join(tmpdir(), 'strands-web-llm-'))
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
    '@strands-agents/sdk@1.21.0',
    '@mlc-ai/web-llm@0.2.86',
    'typescript@6.0.3',
    '@types/node@24.19.2',
  ])
  const audit = JSON.parse(run('npm', ['audit', '--omit=dev', '--json']))
  if (audit.metadata.vulnerabilities.total)
    throw new Error('Published dependencies failed npm audit.')
  const packageRoot = join(temporary, 'node_modules/strands-web-llm')
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
  if (source.includes('@langchain/')) throw new Error('The Strands package depends on LangChain.')
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
import { Agent, FunctionTool } from '@strands-agents/sdk'
import { InMemoryStorage } from '@strands-agents/sdk/storage'
import { MLCEngine } from '@mlc-ai/web-llm'
import {
  BrowserModel,
  WebLLMEmbeddingSearchStrategy,
  WebLLMModel,
  createBrowserBridge,
  withEngine,
} from 'strands-web-llm'

const engine = new MLCEngine({ logLevel: 'SILENT' })
const chunk = (
  delta: Record<string, unknown>,
  finish: string,
) => ({
  id: 'test',
  model: 'test-model',
  object: 'chat.completion.chunk',
  created: 0,
  choices: [{ index: 0, delta, finish_reason: finish }],
})
let turn = 0
engine.chat.completions.create = (async () => {
  turn++
  const items = turn === 1
    ? [
        chunk(
          {
            tool_calls: [
              {
                index: 0,
                id: 'call-1',
                type: 'function',
                function: { name: 'echo', arguments: '{}' },
              },
            ],
          },
          'tool_calls',
        ),
      ]
    : [chunk({ content: 'Finished' }, 'stop')]
  return (async function* () {
    yield* items
  })()
}) as unknown as typeof engine.chat.completions.create

let toolCalls = 0
const agent = new Agent({
  model: new WebLLMModel({ engine, modelId: 'test-model' }),
  tools: [
    new FunctionTool({
      name: 'echo',
      description: 'Echo text',
      callback: () => {
        toolCalls++
        return 'Echoed'
      },
    }),
  ],
  printer: false,
})
assert.match(String(await agent.invoke('Echo this')), /Finished/)
assert.equal(toolCalls, 1)

const strategy = new WebLLMEmbeddingSearchStrategy({
  includeData: false,
  embed: async (texts) => texts.map(() => [1, 0]),
})
const storage = new InMemoryStorage({ searchStrategy: strategy })
await storage.write('note', new TextEncoder().encode('Cloud facts'))
const match = (await storage.search('cloud'))[0]
assert.equal(match?.key, 'note')
assert.equal(match?.data, undefined)

const page = {
  async evaluate<T, A>(_run: (arg: A) => T | Promise<T>, _arg: A): Promise<T> {
    throw new Error('unused')
  },
}
assert.equal(new BrowserModel(page, { modelId: 'test-model' }).getConfig().modelId, 'test-model')
const bridge = createBrowserBridge(engine)
assert.equal(typeof bridge.start, 'function')
await bridge.close()
assert.equal(await withEngine(engine, (current) => current === engine), true)
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
    `strands-web-llm: installed tarball, public types, runtime hashes, Agent tool call, and dependency audit passed (${pack.files.length} files).`,
  )
} finally {
  await rm(temporary, { recursive: true, force: true })
}
