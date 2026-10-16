import { defineConfig } from 'vite'

export default defineConfig({
  resolve: { dedupe: ['@strands-agents/sdk', '@mlc-ai/web-llm'] },
})
