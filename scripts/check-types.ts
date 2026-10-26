import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'

/** Check consumers in both module modes and public declarations with Bundler resolution. */
export function checkTypes(directory: string, packageName: string): void {
  const require = createRequire(join(directory, 'package.json'))
  const ts: typeof import('typescript') = require('typescript')
  const files = [
    join(directory, 'consumer.ts'),
    join(dirname(require.resolve(`${packageName}/runtime/manifest.json`)), '../index.d.ts'),
  ]
  if (typeof ts.createCompilerHost !== 'function') {
    checkNativeTypes(directory, files)
    return
  }
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

function checkNativeTypes(directory: string, files: string[]): void {
  const require = createRequire(join(directory, 'package.json'))
  // The pinned native API checks our declarations without skipping every .d.ts file.
  const {
    API,
  }: typeof import('@typescript/native/unstable/sync') = require('typescript/unstable/sync')
  const temporary = mkdtempSync(join(directory, '.typecheck-'))
  try {
    for (const [module, moduleResolution] of [
      ['ESNext', 'Bundler'],
      ['NodeNext', 'NodeNext'],
    ] as const) {
      const config = join(temporary, `tsconfig.${moduleResolution}.json`)
      writeFileSync(
        config,
        JSON.stringify({
          compilerOptions: {
            noEmit: true,
            strict: true,
            exactOptionalPropertyTypes: true,
            noUncheckedIndexedAccess: true,
            types: ['node'],
            target: 'ES2022',
            module,
            moduleResolution,
          },
          files,
        }),
      )
      const api = new API({ cwd: directory })
      try {
        const snapshot = api.updateSnapshot({ openProjects: [config] })
        const program = snapshot.getProject(config)?.program
        if (!program) throw new Error(`Missing typecheck project: ${config}`)
        const checked = moduleResolution === 'Bundler' ? files : files.slice(0, 1)
        for (const file of files)
          if (!program.getSourceFileMetadata(file))
            throw new Error(`Missing typecheck file: ${file}`)
        const diagnostics = [
          ...program.getConfigFileParsingDiagnostics(),
          ...program.getProgramDiagnostics(),
          ...program.getGlobalDiagnostics(),
          ...program.getSyntacticDiagnostics(),
          ...checked.flatMap((file) => [
            ...program.getBindDiagnostics(file),
            ...program.getSemanticDiagnostics(file),
          ]),
        ]
        if (diagnostics.length) throw new Error(JSON.stringify(diagnostics, null, 2))
      } finally {
        api.close()
      }
    }
  } finally {
    rmSync(temporary, { recursive: true, force: true })
  }
}
