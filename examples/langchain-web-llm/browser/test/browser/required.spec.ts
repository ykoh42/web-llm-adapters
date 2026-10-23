import { createAgent, toolStrategy } from 'langchain'
import { BrowserChatWebLLM, type WebLLMChunk } from 'langchain-web-llm'
import type {
  ChatCompletionRequestStreaming,
  CompletionCreateParamsStreaming,
} from '@mlc-ai/web-llm'
import { writeFile } from 'node:fs/promises'
import { test, expect } from '../../../fixtures/browser.js'
import { sampling } from '../../../sampling.js'

for (const modelId of ['gemma-4-E2B-it-q4f16_1-MLC', 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC']) {
  test(`${modelId} requires one tool and returns an agent response schema`, async ({
    page,
  }, info) => {
    await page.evaluate((id) => window.checks.load(id), modelId)
    await page.evaluate(() => {
      window.requiredToolRequests = []
      const engine = window.checks.engine()
      for (const endpoint of [engine.chat.completions, engine.completions]) {
        endpoint.create = new Proxy(endpoint.create, {
          apply(target, self, args) {
            window.requiredToolRequests.push(structuredClone(args[0]))
            return Reflect.apply(target, self, args)
          },
        })
      }
    })
    const observed: WebLLMChunk[] = []
    const model = new BrowserChatWebLLM(page, {
      modelId,
      request: {
        ...sampling(modelId, modelId.startsWith('gemma-')),
        max_tokens: 4096,
        seed: 7,
      },
      onChunk: (chunk) => observed.push(chunk),
    })
    const results = []
    for (const tool_choice of ['lookup', 'required', 'any']) {
      const result = await model.invoke('Call lookup for code A7.', {
        tool_choice,
        tools: [
          {
            type: 'function',
            function: {
              name: 'lookup',
              description: 'Look up a code.',
              parameters: {
                type: 'object',
                properties: { code: { type: 'string' } },
                required: ['code'],
              },
            },
          },
        ],
      })
      expect(result.tool_calls).toMatchObject([{ name: 'lookup', args: { code: 'A7' } }])
      results.push(result)
    }
    const agent = createAgent({
      model,
      responseFormat: toolStrategy({
        type: 'object',
        properties: { answer: { type: 'number' } },
        required: ['answer'],
      }),
    })
    observed.length = 0
    const result = await agent.invoke({
      messages: [{ role: 'user', content: 'Return answer set to six times seven.' }],
    })
    const request = await page.evaluate(() => window.requiredToolRequests.at(-1)!)
    const native = await page.evaluate(async (request) => {
      const engine = window.checks.engine()
      await engine.resetChat()
      const stream =
        'messages' in request
          ? await engine.chat.completions.create(request)
          : await engine.completions.create(request)
      const chunks: WebLLMChunk[] = []
      for await (const chunk of stream) chunks.push(chunk)
      return chunks
    }, request)
    await writeFile(
      info.outputPath('required-tool.json'),
      JSON.stringify({ results, result, request, observed, native }, null, 2),
    )
    const output = nativeOutput(native)
    expect(nativeOutput(observed)).toEqual(output)
    if (modelId.startsWith('gemma-')) {
      expect(result.structuredResponse).toEqual({ answer: 42 })
    } else {
      // Hermes can return a wrong arithmetic result even without the adapter.
      expect(output.calls).toHaveLength(1)
      expect(result.structuredResponse).toEqual(JSON.parse(output.calls[0]!.input!))
      expect(result.structuredResponse).toEqual({ answer: expect.any(Number) })
    }
  })
}

function nativeOutput(chunks: WebLLMChunk[]) {
  const choices = chunks.flatMap<WebLLMChunk['choices'][number]>((chunk) => chunk.choices)
  return {
    text: choices
      .map((choice) => ('delta' in choice ? (choice.delta.content ?? '') : choice.text))
      .join(''),
    calls: choices
      .flatMap((choice) => ('delta' in choice ? (choice.delta.tool_calls ?? []) : []))
      .map((call) => ({ name: call.function?.name, input: call.function?.arguments })),
    stops: choices.flatMap((choice) => (choice.finish_reason ? [choice.finish_reason] : [])),
  }
}

declare global {
  interface Window {
    requiredToolRequests: (ChatCompletionRequestStreaming | CompletionCreateParamsStreaming)[]
  }
}
