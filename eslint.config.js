import js from '@eslint/js'
import { defineConfig } from 'eslint/config'
import ts from 'typescript-eslint'

export default defineConfig(
  {
    ignores: [
      '**/dist/**',
      '**/node_modules/**',
      '**/coverage/**',
      '.cache/**',
      'test-results/**',
      'playwright-report/**',
    ],
  },
  js.configs.recommended,
  ts.configs.recommended,
  { rules: { 'no-duplicate-imports': 'error' } },
)
