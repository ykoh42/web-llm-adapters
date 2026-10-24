import { configDefaults, defineConfig } from 'vitest/config'

export default defineConfig({
  server: {
    watch: { ignored: ['**/.cache/**', '**/dist/**', '**/test-results/**', '**/coverage/**'] },
  },
  test: {
    globalSetup: ['scripts/test-watch.ts'],
    setupFiles: ['scripts/check-test-build.ts'],
    forceRerunTriggers: [
      ...configDefaults.forceRerunTriggers,
      '**/packages/*/src/**',
      '**/packages/*/tsdown.config.ts',
      '**/shared/**',
    ],
    include: [
      'packages/*/test/**/*.test.ts',
      'examples/*/*/test/**/*.test.ts',
      'test/**/*.test.ts',
    ],
    coverage: {
      provider: 'v8',
      include: ['packages/*/src/**/*.ts', 'shared/web-llm/**/*.ts', 'shared/browser/stream.ts'],
      reporter: ['text', 'html'],
      thresholds: {
        lines: 95,
        branches: 90,
        'packages/langchain-web-llm/src/**': { lines: 95, branches: 90 },
        'packages/pi-ai-web-llm/src/**': { lines: 95, branches: 90 },
        'packages/strands-web-llm/src/**': { lines: 95, branches: 90 },
        'shared/web-llm/**': { lines: 95, branches: 90 },
      },
    },
  },
})
