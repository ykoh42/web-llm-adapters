import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    include: [
      'packages/*/test/**/*.test.ts',
      'examples/*/*/test/**/*.test.ts',
      'test/**/*.test.ts',
    ],
    coverage: {
      provider: 'v8',
      include: ['packages/*/src/**/*.ts', 'shared/web-llm/**/*.ts'],
      reporter: ['text', 'html'],
      thresholds: { lines: 90, branches: 85 },
    },
  },
})
