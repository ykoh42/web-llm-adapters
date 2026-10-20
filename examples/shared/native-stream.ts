import type { MLCEngineInterface } from '@mlc-ai/web-llm'

/** Pass to page.evaluate. Observe completion without changing native chunks. */
export async function watchNativeFinish(app: 'strands' | 'langchain') {
  const page = window as typeof window & {
    checks: { engine(): MLCEngineInterface }
    webllmBridge: { withEngine(run: (engine: MLCEngineInterface) => void): Promise<void> }
  }
  Object.assign(window, { nativeStreamDone: false })
  function patch(engine: MLCEngineInterface) {
    engine.chat.completions.create = new Proxy(engine.chat.completions.create, {
      async apply(target, self, args) {
        const source = await Reflect.apply(target, self, args)
        return (async function* () {
          for await (const chunk of source) {
            if (chunk.choices[0]?.finish_reason) Object.assign(window, { nativeStreamDone: true })
            yield chunk
          }
        })()
      },
    })
  }
  if (app === 'langchain') patch(page.checks.engine())
  else await page.webllmBridge.withEngine(patch)
}
