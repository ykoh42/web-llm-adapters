import { defineConfig } from '@playwright/test'

export default defineConfig({
  outputDir: './test-results/strands-web-llm/extensions',
  testDir: './examples/strands-web-llm/browser/test/extensions',
  timeout: 15 * 60 * 1000,
  workers: 1,
  webServer: [
    {
      command:
        'pnpm --filter @web-llm-adapters/strands-browser exec vite preview --host 127.0.0.1 --port 4173 --strictPort',
      url: 'http://localhost:4173',
      reuseExistingServer: !process.env.CI,
    },
    {
      command:
        'pnpm --filter @web-llm-adapters/strands-browser exec vite preview --host 127.0.0.1 --port 4174 --strictPort',
      url: 'http://localhost:4174',
      reuseExistingServer: !process.env.CI,
    },
  ],
})
