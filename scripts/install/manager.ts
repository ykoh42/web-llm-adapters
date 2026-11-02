import { execFileSync } from 'node:child_process'
import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

interface Consumer {
  manager: 'npm' | 'yarn-pnp' | 'yarn-node-modules'
  directory: string
  name: string
  archive: string
  dependencies: string[]
  source: string
  commonJS: string
}

type Run = (command: string, args: string[], cwd?: string) => string

async function getDependencies(run: Run, name: string, archive: string, dependencies: string[]) {
  const result: Record<string, string> = { [name]: `file:${archive}` }
  for (const dependency of dependencies) {
    const separator = dependency.lastIndexOf('@')
    result[dependency.slice(0, separator)] = dependency.slice(separator + 1)
    const manifest: {
      peerDependencies?: Record<string, string>
      peerDependenciesMeta?: Record<string, { optional?: boolean }>
    } = JSON.parse(run('npm', ['view', dependency, '--json']))
    for (const [peer, range] of Object.entries(manifest.peerDependencies ?? {}))
      if (!manifest.peerDependenciesMeta?.[peer]?.optional) result[peer] ??= range
  }
  return result
}

async function prepareManager(
  manager: Consumer['manager'],
  directory: string,
  consumer: string,
  run: Run,
) {
  if (manager === 'npm') return ''

  const corepack = join(directory, 'tools/node_modules/corepack/dist')
  await mkdir(join(directory, 'tools'))
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
  return corepack
}

function installDependencies(manager: Consumer['manager'], execute: (args: string[]) => string) {
  const devDependencies = ['typescript@5.9.3', '@types/node@26.6.5', 'vite@8.3.5']
  if (manager === 'npm') {
    execute([
      'install',
      '--save-dev',
      '--save-exact',
      '--no-audit',
      '--no-fund',
      ...devDependencies,
    ])
    execute(['ci'])
  } else {
    execute(['add', '--dev', '--exact', ...devDependencies])
    execute(['install', '--immutable'])
  }
}

/** Exercise the packed package outside the repository with each consumer layout. */
export async function checkManagerInstall({
  manager,
  directory,
  name,
  archive,
  dependencies,
  source,
  commonJS,
}: Consumer): Promise<void> {
  const consumer = join(directory, 'consumer')
  await mkdir(consumer)
  const env = { ...process.env, COREPACK_HOME: join(directory, 'corepack') }
  const run = (command: string, args: string[], cwd = consumer) =>
    execFileSync(command, args, { cwd, env, encoding: 'utf8', stdio: 'pipe' })
  const yarn = manager.startsWith('yarn-')
  const cli = yarn ? 'yarn' : manager
  const version =
    cli === 'npm' ? run('npm', ['--version']).trim() : cli === 'yarn' ? '4.18.1' : '12.16.1'
  const packageManager = cli + '@' + version
  const consumerDependencies = await getDependencies(run, name, archive, dependencies)
  const { allowScripts }: { allowScripts: Record<string, boolean> } = JSON.parse(
    await readFile(new URL('../../package.json', import.meta.url), 'utf8'),
  )
  await writeFile(
    join(consumer, 'package.json'),
    JSON.stringify({
      name: 'adapter-consumer',
      private: true,
      type: 'module',
      packageManager,
      allowScripts,
      scripts: {
        smoke: 'node consumer.ts',
        commonjs: 'node commonjs.cjs',
        assets: 'node assets.ts ' + name + ' ' + manager,
        types: 'node check-types.ts ' + name,
        build: 'vite build',
      },
      dependencies: consumerDependencies,
    }),
  )
  await writeFile(join(consumer, 'consumer.ts'), source)
  await writeFile(join(consumer, 'commonjs.cjs'), commonJS)
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

  const corepack = await prepareManager(manager, directory, consumer, run)
  const execute = (args: string[]) => {
    if (manager === 'npm') return run('npm', args)
    const shim = join(corepack, 'yarn.js')
    return run(process.execPath, [shim, ...args])
  }
  if (execute(['--version']).trim() !== version) throw new Error(`Expected ${manager} ${version}.`)

  // Adding dependencies creates the fresh lockfile; the next install must preserve it.
  installDependencies(manager, execute)
  for (const script of ['smoke', 'commonjs', 'assets', 'types', 'build']) execute(['run', script])
  console.log(
    `${name}: ${manager} ${version} install, API, assets, types, and browser build passed.`,
  )
}
