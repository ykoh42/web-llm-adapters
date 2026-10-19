import { defineConfig } from 'vite'

export default defineConfig({
  base: '/langchain/',
  worker: { format: 'es' },
  resolve: { dedupe: ['@langchain/core', '@mlc-ai/web-llm', 'langchain', '@langchain/langgraph'] },
  build: { chunkSizeWarningLimit: 2000 },
})
