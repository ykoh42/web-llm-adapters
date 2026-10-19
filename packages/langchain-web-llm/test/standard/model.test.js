import { MLCEngine } from '@mlc-ai/web-llm'
import { ChatModelUnitTests } from '../../../../.cache/langchain-standard/unit_tests/vitest.js'
import { ChatWebLLM } from '../../src/index.js'

class StandardTests extends ChatModelUnitTests {
  constructor() {
    super({
      Cls: ChatWebLLM,
      constructorArgs: {
        engine: new MLCEngine({ logLevel: 'SILENT' }),
        modelId: 'Qwen3.5-0.8B-q4f16_1-MLC',
        request: { temperature: 0, max_tokens: 100, stop: [] },
      },
      chatModelHasToolCalling: true,
      chatModelHasStructuredOutput: true,
    })
  }
}
new StandardTests().runTests('ChatWebLLM official unit suite')
