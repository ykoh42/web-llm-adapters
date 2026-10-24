import { expect, it } from 'vitest'
import { startupJavaScriptFiles } from '../../scripts/startup-size.ts'

it('counts the static import closure once and skips lazy assets', () => {
  const manifest = {
    'index.html': {
      file: 'index.html',
      imports: ['app.js', 'shared.js'],
      dynamicImports: ['model.js'],
      assets: ['model.wasm'],
    },
    'app.js': { file: 'app.js', imports: ['shared.js', 'styles.css'] },
    'shared.js': { file: 'shared.js' },
    'styles.css': { file: 'styles.css' },
    'model.js': { file: 'model.js' },
  }

  expect([...startupJavaScriptFiles(manifest, 'index.html')]).toEqual(['app.js', 'shared.js'])
})

it('reports a missing static import', () => {
  const manifest = { 'index.html': { file: 'index.html', imports: ['missing.js'] } }

  expect(() => startupJavaScriptFiles(manifest, 'index.html')).toThrow(
    'Startup import is missing from the Vite manifest: missing.js',
  )
})
