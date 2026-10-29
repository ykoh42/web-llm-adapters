import type { MLCEngineInterface } from '@mlc-ai/web-llm'
import { expect, type Page } from '@playwright/test'

type Adapter = 'strands' | 'langchain' | 'pi'

/** Check WebLLM's saved text and continuation after an adapter call is interrupted. */
export async function checkResume(
  page: Page,
  adapter: Adapter,
  sessionId: string,
  partial: string,
) {
  expect(partial.length).toBeGreaterThan(0)
  const recovered = await page.evaluate(
    async ({ adapter, sessionId }) => {
      type Bridge = {
        withEngine<T>(run: (engine: MLCEngineInterface) => Promise<T>): Promise<T>
      }
      const page = window as unknown as {
        webllmBridge: Bridge
        webllmLangChainBridge: Bridge
        piChecks: {
          provider(enabled: boolean): Promise<{ engine: MLCEngineInterface }>
          withEngine<T>(
            engine: MLCEngineInterface,
            run: (engine: MLCEngineInterface) => Promise<T>,
          ): Promise<T>
        }
      }
      async function resume(engine: MLCEngineInterface) {
        try {
          const saved = await engine.resumeChatCompletion(sessionId)
          const result = await engine.resumeChatCompletion(sessionId, { continueGeneration: true })
          if (!('recoveredText' in saved) || !('recoveredText' in result))
            throw new Error('Expected native recovery results.')
          return { saved, result }
        } finally {
          await engine.deleteResumableSession(sessionId)
        }
      }
      if (adapter === 'pi') {
        const { engine } = await page.piChecks.provider(false)
        return page.piChecks.withEngine(engine, resume)
      }
      const bridge = adapter === 'strands' ? page.webllmBridge : page.webllmLangChainBridge
      return bridge.withEngine(resume)
    },
    { adapter, sessionId },
  )

  expect(recovered.saved.recoveredText.startsWith(partial)).toBe(true)
  expect(recovered.result.recoveryMode).toMatch(/^(kv|token_replay)$/)
  expect(recovered.result.emittedTokens).toBeGreaterThan(recovered.saved.emittedTokens)
  expect(recovered.result.recoveredText.startsWith(recovered.saved.recoveredText)).toBe(true)
}
