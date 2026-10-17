import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './examples/strands-web-llm/browser/test/extensions',
  timeout: 15 * 60 * 1000,
  workers: 1,
  webServer: [
    {
      command:
        'npm run build && npm exec --workspace @web-llm-adapters/strands-browser -- vite preview --host 127.0.0.1 --port 4173 --strictPort',
      url: 'http://localhost:4173',
      reuseExistingServer: !process.env.CI,
    },
    {
      command:
        'npm exec --workspace @web-llm-adapters/strands-browser -- vite preview --host 127.0.0.1 --port 4174 --strictPort',
      url: 'http://localhost:4174',
      reuseExistingServer: !process.env.CI,
    },
  ],
})
