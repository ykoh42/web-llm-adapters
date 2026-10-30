import { defineConfig } from '@playwright/test'

export default defineConfig({
  projects: [
    {
      name: 'strands-browser',
      outputDir: './test-results/strands-web-llm/browser',
      testDir: './examples/strands-web-llm/browser/test/browser',
      testMatch: '**/*.spec.ts',
    },
    {
      name: 'strands-harness',
      outputDir: './test-results/strands-web-llm/harness',
      testDir: './examples/strands-web-llm/harness/test/browser',
      testMatch: '**/*.browser.ts',
    },
  ],
  timeout: 15 * 60 * 1000,
  workers: 1,
  webServer: {
    command: 'node scripts/browser/server.ts',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: !process.env.CI,
  },
})
