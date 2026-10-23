import { execFileSync } from 'node:child_process'
import { createRequire } from 'node:module'
import { join } from 'node:path'

/** Check consumers in both module modes and public declarations with Bundler resolution. */
export function checkTypes(directory: string, packageName: string): void {
  const require = createRequire(join(directory, 'package.json'))
  const ts: typeof import('typescript') = require('typescript')
  if (typeof ts.createCompilerHost !== 'function') {
    checkTypesWithCompiler(directory, packageName)
    return
  }

  const files = [
    join(directory, 'consumer.ts'),
    join(directory, 'node_modules', packageName, 'dist/index.d.ts'),
  ]
  for (const [module, moduleResolution] of [
    [ts.ModuleKind.ESNext, ts.ModuleResolutionKind.Bundler],
    [ts.ModuleKind.NodeNext, ts.ModuleResolutionKind.NodeNext],
  ] as const) {
    const options = {
      noEmit: true,
      strict: true,
      exactOptionalPropertyTypes: true,
      noUncheckedIndexedAccess: true,
      types: ['node'],
      target: ts.ScriptTarget.ES2022,
      module,
      moduleResolution,
    }
    const host = ts.createCompilerHost(options)
    host.getCurrentDirectory = () => directory
    const program = ts.createProgram(files, options, host)
    const sources = files.map((path) => {
      const source = program.getSourceFile(path)
      if (!source) throw new Error(`Missing typecheck file: ${path}`)
      return source
    })
    // WebLLM's extensionless type re-exports do not resolve under NodeNext.
    const checked =
      moduleResolution === ts.ModuleResolutionKind.Bundler ? sources : sources.slice(0, 1)
    const diagnostics = [
      ...program.getOptionsDiagnostics(),
      ...program.getGlobalDiagnostics(),
      ...program.getSyntacticDiagnostics(),
      ...checked.flatMap((source) => program.getSemanticDiagnostics(source)),
    ]
    if (diagnostics.length)
      throw new Error(ts.formatDiagnosticsWithColorAndContext(diagnostics, host))
  }
}

function checkTypesWithCompiler(directory: string, packageName: string): void {
  const compiler = join(directory, 'node_modules', 'typescript', 'bin', 'tsc')
  const consumer = join(directory, 'consumer.ts')
  const declaration = join(directory, 'node_modules', packageName, 'dist/index.d.ts')
  const checks = [
    { module: 'ESNext', resolution: 'Bundler', files: [consumer, declaration] },
    { module: 'NodeNext', resolution: 'NodeNext', files: [consumer] },
  ]

  for (const check of checks)
    execFileSync(
      process.execPath,
      [
        compiler,
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
        check.module,
        '--moduleResolution',
        check.resolution,
        ...check.files,
      ],
      { cwd: directory, encoding: 'utf8', stdio: 'pipe' },
    )
}
