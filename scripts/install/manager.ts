import { execFileSync } from 'node:child_process'
import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

interface Consumer {
  manager: 'yarn-pnp' | 'yarn-node-modules' | 'pnpm'
  directory: string
  name: string
  archive: string
  dependencies: string[]
  source: string
}

/** Exercise the packed package outside the repository with each consumer layout. */
export async function checkManagerInstall({
  manager,
  directory,
  name,
  archive,
  dependencies,
  source,
}: Consumer): Promise<void> {
  const consumer = join(directory, 'consumer')
  await mkdir(consumer)
  const env = { ...process.env, COREPACK_HOME: join(directory, 'corepack') }
  const run = (command: string, args: string[], cwd = consumer) =>
    execFileSync(command, args, { cwd, env, encoding: 'utf8', stdio: 'pipe' })
  const yarn = manager.startsWith('yarn-')
  const { packageManager: rootManager }: { packageManager: string } = JSON.parse(
    await readFile(new URL('../../package.json', import.meta.url), 'utf8'),
  )
  const packageManager = yarn ? 'yarn@4.18.1' : rootManager
  const version = packageManager.slice(packageManager.lastIndexOf('@') + 1)
  const consumerDependencies: Record<string, string> = { [name]: `file:${archive}` }
  for (const dependency of dependencies) {
    const separator = dependency.lastIndexOf('@')
    consumerDependencies[dependency.slice(0, separator)] = dependency.slice(separator + 1)
    const manifest: {
      peerDependencies?: Record<string, string>
      peerDependenciesMeta?: Record<string, { optional?: boolean }>
    } = JSON.parse(run('npm', ['view', dependency, '--json']))
    for (const [peer, range] of Object.entries(manifest.peerDependencies ?? {}))
      if (!manifest.peerDependenciesMeta?.[peer]?.optional) consumerDependencies[peer] ??= range
  }
  await writeFile(
    join(consumer, 'package.json'),
    JSON.stringify({
      name: 'adapter-consumer',
      private: true,
      type: 'module',
      packageManager,
      scripts: {
        smoke: 'node consumer.ts',
        assets: `node assets.ts ${name} ${manager}`,
        types: `node check-types.ts ${name}`,
        build: 'vite build',
      },
      dependencies: consumerDependencies,
    }),
  )
  await writeFile(join(consumer, 'consumer.ts'), source)
  await copyFile(new URL('./assets.ts', import.meta.url), join(consumer, 'assets.ts'))
  await writeFile(
    join(consumer, 'check-types.ts'),
    (await readFile(new URL('../check-types.ts', import.meta.url), 'utf8')) +
      `\ncheckTypes(process.cwd(), ${JSON.stringify(name)})\n`,
  )
  await writeFile(join(consumer, 'index.html'), '<script type="module" src="/browser.ts"></script>')
  await writeFile(
    join(consumer, 'browser.ts'),
    `import { MLCEngine } from '@mlc-ai/web-llm'
import * as adapter from '${name}'
import wasm from '${name}/runtime/gemma-4-e2b.wasm?url'
import manifest from '${name}/runtime/manifest.json'
Object.assign(globalThis, { adapter, engine: new MLCEngine(), wasm, manifest })
`,
  )

  const corepack = join(directory, 'tools/node_modules/corepack/dist/yarn.js')
  if (yarn) {
    // Keep tool installation and Corepack's cache outside the consumer and repository.
    run('npm', [
      'install',
      '--prefix',
      join(directory, 'tools'),
      '--no-audit',
      '--no-fund',
      'corepack@0.36.0',
    ])
    if (manager === 'yarn-node-modules')
      await writeFile(join(consumer, '.yarnrc.yml'), 'nodeLinker: node-modules\n')
  } else {
    // These dependency hooks are optional setup, not required to use their libraries.
    await writeFile(
      join(consumer, 'pnpm-workspace.yaml'),
      "allowBuilds:\n  '@google/genai': false\n  protobufjs: false\n",
    )
  }
  const execute = (args: string[]) =>
    yarn ? run(process.execPath, [corepack, ...args]) : run('pnpm', args)
  if (execute(['--version']).trim() !== version) throw new Error(`Expected ${manager} ${version}.`)

  // Adding dependencies creates the fresh lockfile; the next install must preserve it.
  execute(['add', '-D', '-E', 'typescript@5.9.3', '@types/node@26.6.5', 'vite@8.3.5'])
  execute(['install', yarn ? '--immutable' : '--frozen-lockfile'])
  for (const script of ['smoke', 'assets', 'types', 'build']) execute(['run', script])
  console.log(
    `${name}: ${manager} ${version} install, API, assets, types, and browser build passed.`,
  )
}
