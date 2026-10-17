import { execFileSync } from 'node:child_process'
import { copyFile, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../../../', import.meta.url))
const npm = process.env.npm_execpath
if (!npm) throw new Error('Run this check with npm run test:install.')
const directory = await mkdtemp(join(tmpdir(), 'webllm-install-'))

function run(command: string, args: string[]) {
  execFileSync(command, args, { cwd: directory, stdio: 'inherit' })
}

try {
  const library = JSON.parse(
    await readFile(join(root, 'packages/strands-webllm/package.json'), 'utf8'),
  )
  const example = JSON.parse(await readFile(new URL('../package.json', import.meta.url), 'utf8'))
  const project = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'))
  console.log('Pack the library and install it outside the workspace.')
  const packed = JSON.parse(
    execFileSync(
      process.execPath,
      [npm, 'pack', '--workspace', 'strands-webllm', '--pack-destination', directory, '--json'],
      { cwd: root, encoding: 'utf8' },
    ),
  )
  await writeFile(
    join(directory, 'package.json'),
    JSON.stringify({
      private: true,
      type: 'module',
      dependencies: {
        'strands-webllm': `file:${join(directory, packed[0].filename)}`,
        '@mlc-ai/web-llm': library.devDependencies['@mlc-ai/web-llm'],
        '@strands-agents/sdk': library.devDependencies['@strands-agents/sdk'],
        '@strands-agents/harness': example.dependencies['@strands-agents/harness'],
        '@types/node': project.devDependencies['@types/node'],
      },
    }),
  )
  run(process.execPath, [
    npm,
    'install',
    '--ignore-scripts',
    '--no-audit',
    '--no-fund',
    '--package-lock=false',
  ])
  console.log('Check browser and Node.js public types.')
  for (const [file, module, moduleResolution] of [
    ['consumer.ts', 'ESNext', 'Bundler'],
    ['node.ts', 'NodeNext', 'NodeNext'],
  ] as const) {
    await copyFile(new URL('./install/' + file, import.meta.url), join(directory, file))
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
      join(root, 'node_modules/typescript/bin/tsc'),
      '--project',
      'tsconfig.json',
    ])
  }
  console.log('Run the installed package.')
  run(process.execPath, ['--experimental-strip-types', '--test', 'consumer.ts'])
} finally {
  await rm(directory, { recursive: true, force: true })
}
