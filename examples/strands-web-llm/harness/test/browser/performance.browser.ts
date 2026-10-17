import { writeFile } from 'node:fs/promises'
import { Agent, Message, TextBlock } from '@strands-agents/sdk'
import { BrowserModel, type BrowserPage } from 'strands-web-llm'
import { test, expect } from '../../../fixtures/browser.js'

type Mode = 'native' | 'provider' | 'bridge'
type Sample = {
  ms: number
  firstTokenMs: number
  tokens: number
  text: string
  calls: number
  trace: unknown
}

function computeMs(sample: Sample) {
  const trace = sample.trace
  if (
    !trace ||
    typeof trace !== 'object' ||
    !('time_to_first_token_s' in trace) ||
    !('time_per_output_token_s' in trace)
  )
    throw new Error('Missing native timings.')
  const prefill = trace.time_to_first_token_s
  const decode = trace.time_per_output_token_s
  if (typeof prefill !== 'number' || typeof decode !== 'number')
    throw new Error('Invalid native timings.')
  return 1000 * (prefill + decode * (sample.tokens - 1))
}

test('performance: compare native, in-page provider, and Node.js bridge on identical inference', async ({
  page,
}, info) => {
  const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
  const prompt = 'List the integers from 1 to 500, separated by commas. No explanation.'
  const config = {
    modelId,
    maxTokens: 128,
    temperature: 0,
    request: { seed: 7, ignore_eos: true, extra_body: { enable_thinking: false } },
  }
  let calls = 0
  const transport: BrowserPage = page
  const endpoint: BrowserPage = {
    evaluate: (run, arg) => {
      calls++
      return transport.evaluate(run, arg)
    },
  }
  const model = new BrowserModel(endpoint, config)
  const messages = [new Message({ role: 'user', content: [new TextBlock(prompt)] })]

  async function measure(mode: Mode): Promise<Sample> {
    await page.evaluate(
      (model) => window.webllmBridge.withEngine((engine) => engine.resetChat(false, model)),
      modelId,
    )
    calls = 0
    const start = performance.now()
    if (mode === 'bridge') {
      let text = ''
      let tokens = 0
      let firstTokenMs = 0
      let trace: unknown
      for await (const event of model.stream(messages)) {
        if (event.type === 'modelContentBlockDeltaEvent' && event.delta.type === 'textDelta') {
          firstTokenMs ||= performance.now() - start
          text += event.delta.text
        }
        if (event.type === 'modelMetadataEvent') {
          tokens = event.usage?.outputTokens ?? 0
          trace = event.trace
        }
      }
      return { ms: performance.now() - start, firstTokenMs, tokens, text, calls, trace }
    }
    const result = await page.evaluate(
      async ({ mode, modelId, prompt, config }) => {
        const start = performance.now()
        let text = ''
        let tokens = 0
        let firstTokenMs = 0
        let trace: unknown
        if (mode === 'native') {
          await window.webllmBridge.withEngine(async (engine) => {
            const stream = await engine.chat.completions.create({
              model: modelId,
              messages: [{ role: 'user', content: prompt }],
              stream: true,
              stream_options: { include_usage: true },
              n: 1,
              max_tokens: 128,
              temperature: 0,
              seed: 7,
              ignore_eos: true,
            })
            for await (const chunk of stream) {
              const content = chunk.choices[0]?.delta.content
              if (content) {
                firstTokenMs ||= performance.now() - start
                text += content
              }
              if (chunk.usage) {
                tokens = chunk.usage.completion_tokens
                trace = chunk.usage.extra
              }
            }
          })
        } else {
          const id = await window.webllmBridge.start({
            messages: [{ role: 'user', content: [{ text: prompt }] }],
            config,
            options: {},
          })
          try {
            for (;;) {
              const result = await window.webllmBridge.next(id)
              if ('error' in result) throw new Error(result.error.message)
              if (result.done) break
              const event = result.value
              if (
                event.type === 'modelContentBlockDeltaEvent' &&
                event.delta.type === 'textDelta'
              ) {
                firstTokenMs ||= performance.now() - start
                text += event.delta.text
              }
              if (event.type === 'modelMetadataEvent') {
                tokens = event.usage?.outputTokens ?? 0
                trace = event.trace
              }
            }
          } finally {
            await window.webllmBridge.stop(id)
          }
        }
        return { firstTokenMs, tokens, text, trace }
      },
      { mode, modelId, prompt, config },
    )
    return { ...result, ms: performance.now() - start, calls: 1 }
  }

  await page.goto('http://127.0.0.1:4173/harness.html')
  await page.waitForFunction(() => Boolean(window.harness))
  await page.evaluate((model) => window.harness.open(model, 2048), modelId)
  try {
    // Warm each path, then rotate order to reduce initialization and thermal bias.
    for (let round = 0; round < 2; round++)
      for (const mode of ['native', 'provider', 'bridge'] as const) await measure(mode)
    const samples: Record<Mode, Sample[]> = { native: [], provider: [], bridge: [] }
    for (const order of [
      ['native', 'provider', 'bridge'],
      ['provider', 'bridge', 'native'],
      ['bridge', 'native', 'provider'],
      ['native', 'bridge', 'provider'],
      ['bridge', 'provider', 'native'],
      ['provider', 'native', 'bridge'],
    ] as const) {
      for (const mode of order) samples[mode].push(await measure(mode))
    }
    const expected = samples.native[0]!
    for (const sample of Object.values(samples).flat()) {
      expect(sample.tokens).toBe(128)
      expect(sample.text).toBe(expected.text)
      expect(computeMs(sample)).toBeGreaterThan(0)
    }
    const median = (values: number[]) => {
      const sorted = [...values].sort((a, b) => a - b)
      const middle = Math.floor(sorted.length / 2)
      return sorted.length % 2 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2
    }
    const nativeMs = median(samples.native.map((sample) => sample.ms))
    const summary = Object.entries(samples).map(([mode, values]) => ({
      mode,
      ms: Math.round(median(values.map((sample) => sample.ms))),
      computeMs: Math.round(median(values.map(computeMs))),
      outsideMs: Math.round(median(values.map((sample) => sample.ms - computeMs(sample)))),
      firstTokenMs: Math.round(median(values.map((sample) => sample.firstTokenMs))),
      overhead: (100 * (median(values.map((sample) => sample.ms)) / nativeMs - 1)).toFixed(1) + '%',
      calls: median(values.map((sample) => sample.calls)),
    }))
    const prefill: number[] = []
    const cached = new BrowserModel(page, {
      modelId,
      maxTokens: 64,
      temperature: 0,
      request: { extra_body: { enable_thinking: false } },
      onChunk: (chunk) => {
        if (chunk.usage) prefill.push(chunk.usage.prompt_tokens)
      },
    })
    const agent = new Agent({ model: cached, printer: false })
    const first = await agent.invoke(
      'Read these notes.\n' + 'The project keeps notes.\n'.repeat(128) + 'Reply ready.',
    )
    expect(first.stopReason).toBe('endTurn')
    await agent.invoke('Reply ready again.')
    expect(prefill).toHaveLength(2)
    expect(prefill[0]).toBeGreaterThan(500)
    expect(prefill[1]).toBeLessThan(prefill[0]! / 4)
    console.table(summary)
    console.log('KV cache prefill tokens:', prefill)
    const path = info.outputPath('performance.json')
    await writeFile(path, JSON.stringify({ summary, samples, prefill }, null, 2))
    await info.attach('performance', { path, contentType: 'application/json' })
  } finally {
    await page.evaluate(() => window.harness.close())
  }
})
