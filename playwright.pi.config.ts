import { defineConfig } from '@playwright/test'

export default defineConfig({
  outputDir: './test-results/pi-ai-web-llm',
  testDir: './examples/pi-ai-web-llm/browser/test',
  testMatch: '**/*.spec.ts',
  timeout: 15 * 60 * 1000,
  workers: 1,
  webServer: {
    command: 'node scripts/browser/server.ts',
    url: 'http://127.0.0.1:4173/pi/',
    reuseExistingServer: !process.env.CI,
  },
})
