import { execFileSync } from 'node:child_process'
import { copyFile, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../../../../../', import.meta.url))
const directory = await mkdtemp(join(tmpdir(), 'webllm-install-'))

function run(command: string, args: string[]) {
  execFileSync(command, args, { cwd: directory, stdio: 'inherit' })
}

try {
  const library = JSON.parse(
    await readFile(join(root, 'packages/strands-web-llm/package.json'), 'utf8'),
  )
  const example = JSON.parse(await readFile(new URL('../../package.json', import.meta.url), 'utf8'))
  const project = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'))
  console.log('Pack the library and install it outside the workspace.')
  const packed = JSON.parse(
    execFileSync(
      'pnpm',
      ['--filter', 'strands-web-llm', 'pack', '--pack-destination', directory, '--json'],
      { cwd: root, encoding: 'utf8' },
    ),
  )
  await writeFile(
    join(directory, 'package.json'),
    JSON.stringify({
      private: true,
      type: 'module',
      dependencies: {
        'strands-web-llm': `file:${resolve(directory, packed.filename)}`,
        '@mlc-ai/web-llm': library.devDependencies['@mlc-ai/web-llm'],
        '@strands-agents/sdk': library.devDependencies['@strands-agents/sdk'],
        '@strands-agents/harness': example.dependencies['@strands-agents/harness'],
        '@types/node': project.devDependencies['@types/node'],
      },
    }),
  )
  run('npm', ['install', '--ignore-scripts', '--no-audit', '--no-fund', '--package-lock=false'])
  console.log('Verify the installed runtime asset, hash, source patch and notices.')
  run(process.execPath, [
    '--input-type=module',
    '-e',
    `import { readFile } from 'node:fs/promises'
     import { createHash } from 'node:crypto'
     import assert from 'node:assert/strict'
     const url = import.meta.resolve('strands-web-llm/runtime/gemma-4-e2b.wasm')
     const manifest = JSON.parse(await readFile(new URL('manifest.json', url), 'utf8'))
     const bytes = await readFile(new URL(url))
     assert.equal(createHash('sha256').update(bytes).digest('hex'), manifest.files['gemma-4-e2b.wasm'])
     assert.ok(WebAssembly.validate(bytes))
     for (const prefix of ['/Users/', '/home/', '\\\\Users\\\\']) {
       assert.ok(!bytes.includes(Buffer.from(prefix)))
     }
     for (const name of ['cache.patch', 'projection.patch', 'model.json', 'build.py', 'check-cache.cc', 'NOTICE.txt', 'licenses/Apache-2.0.txt']) {
       assert.ok((await readFile(new URL(name, url))).length)
     }`,
  ])
  console.log('Check browser and Node.js public types.')
  for (const [file, module, moduleResolution] of [
    ['consumer.ts', 'ESNext', 'Bundler'],
    ['node.ts', 'NodeNext', 'NodeNext'],
  ] as const) {
    await copyFile(new URL('./' + file, import.meta.url), join(directory, file))
    await writeFile(
      join(directory, 'tsconfig.json'),
      JSON.stringify({
        compilerOptions: {
          target: 'ES2022',
          module,
          moduleResolution,
          types: ['node'],
          strict: true,
          noEmit: true,
          skipLibCheck: true,
        },
        include: [file],
      }),
    )
    run(process.execPath, [
      fileURLToPath(import.meta.resolve('typescript/bin/tsc6')),
      '--project',
      'tsconfig.json',
    ])
  }
  console.log('Run the installed package.')
  run(process.execPath, ['--test', 'consumer.ts'])
} finally {
  await rm(directory, { recursive: true, force: true })
}
