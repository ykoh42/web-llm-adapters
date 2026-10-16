import { defineConfig } from 'vite'
import { fileURLToPath } from 'node:url'

export default defineConfig({
  resolve: { dedupe: ['@strands-agents/sdk', '@mlc-ai/web-llm'] },
  build: {
    rolldownOptions: {
      input: {
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
        engine: fileURLToPath(new URL('./engine.html', import.meta.url)),
      },
    },
  },
  server: { headers: { 'Service-Worker-Allowed': '/' } },
  worker: { format: 'es', rolldownOptions: { output: { entryFileNames: '[name]-[hash].js' } } },
})
