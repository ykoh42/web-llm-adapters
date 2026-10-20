import { defineConfig } from 'vite'
import { fileURLToPath } from 'node:url'

export default defineConfig({
  resolve: { dedupe: ['@strands-agents/sdk', '@mlc-ai/web-llm'] },
  build: {
    rolldownOptions: {
      input: {
        'extension-worker': fileURLToPath(new URL('./src/engine/extension.ts', import.meta.url)),
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
        harness: fileURLToPath(new URL('./harness.html', import.meta.url)),
        engine: fileURLToPath(new URL('./test/browser/engine.html', import.meta.url)),
      },
      output: {
        entryFileNames: (chunk) =>
          chunk.name === 'extension-worker' ? 'extension-worker.js' : 'assets/[name]-[hash].js',
      },
    },
  },
  server: { headers: { 'Service-Worker-Allowed': '/' } },
  worker: { format: 'es', rolldownOptions: { output: { entryFileNames: '[name]-[hash].js' } } },
})
