import { execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { runtimeFiles } from '../shared/build.ts'
import { checkManagerInstall } from './install/manager.ts'

interface PackedFile {
  path: string
}

interface InstallCheck {
  root: URL
  peerImports: string[]
  forbiddenImports: string[]
  source: string
  compatibilityDependencies: string[][]
  consumerDependencies?: string[]
}

const consumerTypeScriptVersions = ['5.9.3', '6.0.3', '7.0.2']

/** Check a fresh consumer installation without relying on workspace dependencies. */
export async function checkInstall({
  root,
  peerImports,
  forbiddenImports,
  source,
  compatibilityDependencies,
  consumerDependencies = [],
}: InstallCheck): Promise<void> {
  const manager = process.argv[2]
  if (
    manager !== undefined &&
    manager !== 'yarn-pnp' &&
    manager !== 'yarn-node-modules' &&
    manager !== 'pnpm'
  )
    throw new Error(`Unknown consumer package manager: ${manager}`)
  const directory = fileURLToPath(root)
  const { name }: { name: string } = JSON.parse(
    await readFile(join(directory, 'package.json'), 'utf8'),
  )
  const temporary = await mkdtemp(join(tmpdir(), `${name}-`))
  const run = (cmd: string, args: string[], cwd = temporary) =>
    execFileSync(cmd, args, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] })
  try {
    const pack: { filename: string; files: PackedFile[] } = JSON.parse(
      run(
        'pnpm',
        ['--ignore-scripts', 'pack', '--json', '--pack-destination', temporary],
        directory,
      ),
    )
    checkFiles(pack.files)
    if (manager) {
      const dependencies = compatibilityDependencies.at(-1)
      if (!dependencies) throw new Error('Missing consumer dependencies.')
      await checkManagerInstall({
        manager,
        directory: temporary,
        name,
        archive: resolve(temporary, pack.filename),
        dependencies,
        source,
      })
      return
    }
    await writeFile(
      join(temporary, 'package.json'),
      JSON.stringify({ private: true, type: 'module' }),
    )
    const packageRoot = join(temporary, 'node_modules', name)
    await writeFile(join(temporary, 'consumer.ts'), source)

    for (const dependencies of compatibilityDependencies) {
      run('npm', [
        'install',
        '--ignore-scripts',
        '--no-fund',
        '--no-audit',
        resolve(temporary, pack.filename),
        ...dependencies,
      ])
      await checkDependencyVersions(temporary, dependencies)
      await checkPrivateFiles(packageRoot, pack.files)
      const installed = await readFile(join(packageRoot, 'dist/index.js'), 'utf8')
      for (const other of forbiddenImports)
        if (installed.includes(other)) throw new Error(`${name} depends on another framework.`)
      await checkRuntime(packageRoot)
      run(process.execPath, [
        '--input-type=module',
        '--eval',
        [name, ...peerImports, '@mlc-ai/web-llm']
          .map((specifier) => `await import(${JSON.stringify(specifier)})`)
          .join(';'),
      ])
      run(process.execPath, ['--experimental-strip-types', 'consumer.ts'])
      const audit = JSON.parse(run('npm', ['audit', '--omit=dev', '--json']))
      if (audit.metadata.vulnerabilities.total)
        throw new Error('Published dependencies failed npm audit.')
      console.log(`${name}: compatibility install and dependency audit passed.`)
    }

    for (const version of consumerTypeScriptVersions) {
      run('npm', [
        'install',
        '--save-dev',
        '--ignore-scripts',
        '--no-fund',
        '--no-audit',
        `typescript@${version}`,
        '@types/node@26.6.5',
        ...consumerDependencies,
      ])
      const installedTypeScript: { version: string } = JSON.parse(
        await readFile(join(temporary, 'node_modules/typescript/package.json'), 'utf8'),
      )
      if (installedTypeScript.version !== version)
        throw new Error(`Expected TypeScript ${version}, found ${installedTypeScript.version}.`)
      const checker = new URL('./check-types.ts', import.meta.url)
      run(process.execPath, [
        '--experimental-strip-types',
        '--input-type=module',
        '--eval',
        `import { checkTypes } from ${JSON.stringify(checker.href)}; checkTypes(${JSON.stringify(temporary)}, ${JSON.stringify(name)});`,
      ])
      console.log(`${name}: public types passed TypeScript ${version}.`)
    }
    run(process.execPath, ['--experimental-strip-types', 'consumer.ts'])
    console.log(
      `${name}: installed tarball, public types, runtime hashes, and runtime smoke passed (${pack.files.length} files).`,
    )
  } finally {
    await rm(temporary, { recursive: true, force: true })
  }
}

async function checkDependencyVersions(directory: string, dependencies: string[]): Promise<void> {
  for (const dependency of dependencies) {
    const separator = dependency.lastIndexOf('@')
    if (separator < 1) continue
    const name = dependency.slice(0, separator)
    const expected = dependency.slice(separator + 1)
    const packageFile = join(directory, 'node_modules', ...name.split('/'), 'package.json')
    const installed: { version: string } = JSON.parse(await readFile(packageFile, 'utf8'))
    if (installed.version !== expected)
      throw new Error(`Expected ${name} ${expected}, found ${installed.version}.`)
  }
}

function checkFiles(files: PackedFile[]): void {
  const allowed = new Set([
    'README.md',
    'LICENSE',
    'CHANGELOG.md',
    'package.json',
    'dist/index.js',
    'dist/index.d.ts',
    ...runtimeFiles.map((name) => 'dist/runtime/' + name),
  ])
  const packed = new Set(files.map(({ path }) => path))
  for (const path of allowed) if (!packed.has(path)) throw new Error(`Missing npm file: ${path}`)
  for (const { path } of files)
    if (!allowed.has(path)) throw new Error(`Unexpected npm file: ${path}`)
}

async function checkPrivateFiles(directory: string, files: PackedFile[]): Promise<void> {
  for (const { path } of files) {
    const bytes = await readFile(join(directory, path))
    for (const prefix of [
      '/Users/',
      '/home/',
      'npm_' + 'token=',
      '-----BEGIN ' + 'PRIVATE KEY-----',
    ])
      if (bytes.includes(Buffer.from(prefix))) throw new Error(`Private data in npm file: ${path}`)
  }
}

async function checkRuntime(directory: string): Promise<void> {
  const runtime = join(directory, 'dist/runtime')
  if (((await stat(join(runtime, 'gemma-4-e2b.wasm'))).mode & 0o111) !== 0)
    throw new Error('Wasm package file must not be executable.')
  const manifest = JSON.parse(await readFile(join(runtime, 'manifest.json'), 'utf8'))
  for (const [name, hash] of Object.entries(manifest.files)) {
    const bytes = await readFile(join(runtime, name))
    if (createHash('sha256').update(bytes).digest('hex') !== hash)
      throw new Error('Runtime hash mismatch: ' + name)
  }
  if (!WebAssembly.validate(await readFile(join(runtime, 'gemma-4-e2b.wasm'))))
    throw new Error('Invalid Wasm module.')
}
