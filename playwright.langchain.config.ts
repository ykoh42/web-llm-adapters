import { defineConfig } from '@playwright/test'

export default defineConfig({
  outputDir: './test-results/langchain-web-llm',
  testDir: './examples/langchain-web-llm/browser/test/browser',
  testMatch: '**/*.spec.ts',
  timeout: 15 * 60 * 1000,
  workers: 1,
  webServer: {
    command: 'node scripts/browser/server.ts',
    url: 'http://127.0.0.1:4173/langchain/',
    reuseExistingServer: !process.env.CI,
  },
})
