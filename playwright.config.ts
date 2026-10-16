import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './test/browser',
  timeout: 15 * 60 * 1000,
  workers: 1,
  webServer: {
    command: 'npm run build && npm run preview',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: !process.env.CI,
  },
})
