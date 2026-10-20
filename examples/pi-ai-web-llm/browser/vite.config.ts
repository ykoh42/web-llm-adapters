import { defineConfig } from 'vite'
import { fileURLToPath } from 'node:url'

export default defineConfig({
  base: '/pi/',
  worker: { format: 'es' },
  resolve: {
    dedupe: ['@earendil-works/pi-ai', '@earendil-works/pi-agent-core', '@mlc-ai/web-llm'],
  },
  build: {
    chunkSizeWarningLimit: 2000,
    rolldownOptions: {
      input: {
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
        checks: fileURLToPath(new URL('./test/checks.html', import.meta.url)),
      },
    },
  },
})
