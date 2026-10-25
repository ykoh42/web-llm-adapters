import type { MLCEngineInterface } from '@mlc-ai/web-llm'

/** Pass directly to page.evaluate; all browser work stays inside this function. */
export async function breakToolStream({
  app,
  call,
  after = '',
}: {
  app: 'strands' | 'langchain'
  call: string
  after?: string
}) {
  const page = window as typeof window & {
    checks: { engine(): MLCEngineInterface }
    webllmBridge: { withEngine(run: (engine: MLCEngineInterface) => void): Promise<void> }
  }
  function patch(engine: MLCEngineInterface) {
    let fail = true
    for (const endpoint of [engine.chat.completions, engine.completions]) {
      Object.assign(endpoint, {
        create: new Proxy(endpoint.create, {
          async apply(target, self, args) {
            const source = await Reflect.apply(target, self, args)
            if (!fail) return source
            fail = false
            let body = ''
            return (async function* () {
              for await (const chunk of source) {
                const choice = chunk.choices[0]
                if (choice) body += 'delta' in choice ? (choice.delta.content ?? '') : choice.text
                yield chunk
                const at = after ? body.lastIndexOf(after) : 0
                if (at !== -1 && body.includes(call, at + after.length))
                  throw new Error('Injected native stream failure')
              }
            })()
          },
        }),
      })
    }
  }
  if (app === 'langchain') patch(page.checks.engine())
  else await page.webllmBridge.withEngine(patch)
}

export function expectedOutput(native: string, modelId: string, prefilled = false) {
  if (prefilled) native = '<think>' + native
  const thought = modelId.startsWith('gemma-')
    ? /<\|channel>thought\n([\s\S]*?)<channel\|>/g
    : /<think>([\s\S]*?)<\/think>/g
  const text = native.replace(thought, '')
  return {
    // Interrupted Qwen tool syntax is retained in raw chunks, not answer text.
    text: modelId.startsWith('Qwen') ? text.split('<tool_call>')[0]! : text,
    reasoning: [...native.matchAll(thought)].map((part) => part[1]).join(''),
  }
}
