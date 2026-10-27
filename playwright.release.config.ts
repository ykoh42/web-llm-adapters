import { defineConfig } from '@playwright/test'

export default defineConfig({
  projects: [
    {
      name: 'strands',
      testDir: './examples/strands-web-llm/harness/test/browser',
      testMatch: '**/*.browser.ts',
    },
    {
      name: 'langchain',
      testDir: './examples/langchain-web-llm/browser/test/browser',
      testMatch: '**/*.spec.ts',
    },
    { name: 'pi', testDir: './examples/pi-ai-web-llm/browser/test', testMatch: '**/*.spec.ts' },
    { name: 'integration', testDir: './test/browser', testMatch: '**/*.browser.ts' },
  ],
  grep: /@release/,
  globalSetup: './scripts/release/setup.ts',
  forbidOnly: true,
  retries: 0,
  workers: 1,
  timeout: 15 * 60 * 1000,
  outputDir: './test-results/release',
  reporter: [['list'], ['./scripts/release/reporter.ts']],
  webServer: {
    command: 'node shared/browser/server.ts',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: false,
  },
})
