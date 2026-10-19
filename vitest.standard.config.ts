import { defineConfig } from 'vitest/config'

export default defineConfig({
  root: import.meta.dirname,
  test: {
    include: [
      process.env.REAL_MODELS
        ? 'packages/langchain-web-llm/test/standard/*.browser.test.js'
        : 'packages/langchain-web-llm/test/standard/model.test.js',
    ],
  },
})
