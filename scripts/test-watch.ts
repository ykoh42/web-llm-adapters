import { execFile } from 'node:child_process'
import { glob } from 'node:fs/promises'
import { join, sep } from 'node:path'
import { promisify } from 'node:util'
import type { TestProject } from 'vitest/node'

const run = promisify(execFile)

export default async function setup(project: TestProject): Promise<void> {
  project.provide('adapterBuildError', undefined)
  if (!project.vitest.config.watch) return
  const build = async () => {
    const root = project.config.root
    try {
      await run(process.execPath, ['--run', 'build:adapters'], { cwd: root })
      project.provide('adapterBuildError', undefined)
    } catch (error) {
      // Keep watching after build errors without testing stale code.
      project.provide('adapterBuildError', error instanceof Error ? error.message : String(error))
    }
    for await (const file of glob('packages/*/dist/**/*.{js,mjs,json}', { cwd: root }))
      project.vitest.invalidateFile(join(root, file).replaceAll(sep, '/'))
  }
  let changed = false
  let building: Promise<void> | undefined
  // Finish the latest save before a rerun imports package bundles.
  function rebuild() {
    changed = true
    building ??= (async () => {
      while (changed) {
        changed = false
        await build()
      }
    })().finally(() => {
      building = undefined
    })
    return building
  }
  await rebuild()
  project.onTestsRerun(rebuild)
}
