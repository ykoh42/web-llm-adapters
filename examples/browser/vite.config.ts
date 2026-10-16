import { defineConfig } from 'vite'

export default defineConfig({
  resolve: { dedupe: ['@strands-agents/sdk', '@mlc-ai/web-llm'] },
  server: { headers: { 'Service-Worker-Allowed': '/' } },
  worker: { format: 'es', rolldownOptions: { output: { entryFileNames: '[name]-[hash].js' } } },
})
