import { writeFile } from 'node:fs/promises'
import { AIMessage, HumanMessage, ToolMessage } from '@langchain/core/messages'
import type { CompletionCreateParamsStreaming } from '@mlc-ai/web-llm'
import { BrowserChatWebLLM } from 'langchain-web-llm'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

type Tree = number | { next: Tree }

declare global {
  interface Window {
    gemmaDepthRequests: CompletionCreateParamsStreaming[]
  }
}

let tree: Tree = 42
for (let level = 0; level < 3000; level++) tree = { next: tree }
const cases = [
  {
    title: 'Gemma preserves deep tool values through native thinking and a final answer',
    input: { tree, code: 'A7' },
    literal: '{code:<|"|>A7<|"|>,tree:' + '{next:'.repeat(3000) + '42' + '}'.repeat(3001),
  },
  {
    title: 'Gemma matches canonical mixed-case and Unicode order through real WebLLM',
    input: { B: { '𐐀': 'supplementary', ｚ: 'wide' }, a: 3, A: 4, b: 5, z: 1 },
    literal: '{a:3,A:4,B:{ｚ:<|"|>wide<|"|>,𐐀:<|"|>supplementary<|"|>},b:5,z:1}',
  },
]

for (const { title, input, literal } of cases) {
  test(title, async ({ page }, info) => {
    const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
    const prompt =
      '<|turn>system\n<|think|>\n<turn|>\n' +
      '<|turn>user\nUse lookup.<turn|>\n<|turn>model\n' +
      '<|tool_call>call:lookup' +
      literal +
      '<tool_call|>' +
      '<|tool_response>response:lookup{value:<|"|>Done.<|"|>}<tool_response|>' +
      '<turn|>\n<|turn>user\nWhat is 17 * 23?<turn|>\n<|turn>model\n'
    const settings = sampling(modelId, true)
    // Raw Gemma requests encode thinking in the prompt.
    delete settings.extra_body
    const request: CompletionCreateParamsStreaming = {
      ...settings,
      model: modelId,
      prompt,
      max_tokens: 4096,
      seed: 7,
      stream: true,
      stream_options: { include_usage: true },
      n: 1,
    }
    await page.evaluate((id) => window.checks.load(id), modelId)
    await page.evaluate(
      (modelId) =>
        window.webllmLangChainBridge!.withEngine(async (engine) => {
          await engine.reload(modelId, { context_window_size: 32768 })
          window.gemmaDepthRequests = []
          const endpoint = engine.completions
          endpoint.create = new Proxy(endpoint.create, {
            apply(target, self, args) {
              window.gemmaDepthRequests.push(structuredClone(args[0]))
              return Reflect.apply(target, self, args)
            },
          })
        }),
      modelId,
    )
    let observed = ''
    const model = new BrowserChatWebLLM(page, {
      modelId,
      request: { ...sampling(modelId, true), max_tokens: 4096, seed: 7 },
      onChunk(chunk) {
        const choice = chunk.choices[0]
        if (choice && 'text' in choice) observed += choice.text
      },
    })
    const result = await model.invoke([
      new HumanMessage('Use lookup.'),
      new AIMessage({ content: '', tool_calls: [{ id: 'call', name: 'lookup', args: input }] }),
      new ToolMessage({ tool_call_id: 'call', content: 'Done.' }),
      new HumanMessage('What is 17 * 23?'),
    ])
    const [prepared] = await page.evaluate(() => window.gemmaDepthRequests)
    const reasoning = result.contentBlocks
      .filter((part) => part.type === 'reasoning')
      .map((part) => part.reasoning)
      .join('')
    const evidence = info.outputPath('gemma-depth.json')
    await writeFile(
      evidence,
      JSON.stringify(
        { prepared, expected: request, observed, reasoning, text: result.text },
        null,
        2,
      ),
    )
    await info.attach('gemma-depth', { path: evidence, contentType: 'application/json' })
    expect(prepared).toEqual(request)
    expect(reasoning.trim()).not.toBe('')
    expect(reasoning).toBe(/<\|channel>thought\n([\s\S]*?)<channel\|>/.exec(observed)?.[1])
    expect(result.text).toContain('391')
    expect(result.response_metadata.finish_reason).toBe('stop')
  })
}
