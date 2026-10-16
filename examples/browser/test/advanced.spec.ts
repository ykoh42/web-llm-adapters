import { test, expect, load, run, unload } from './helpers.js'

test('structural tags constrain the JSON inside an answer', async ({ page }) => {
  await load(page, 'tags')
  await run(page)
  const text = (await page.getByLabel('Response').textContent())!
  const body = text.match(/<answer>(.*?)<\/answer>/s)?.[1]
  expect(body).toBeDefined()
  expect(JSON.parse(body!)).toEqual({ count: 3 })
  await unload(page)
})

test('Qwen enables and disables thinking through Strands', async ({ page }) => {
  await load(page, 'thinking')
  await run(page)
  const reports: { thinking?: { enabled: boolean; text: string }[] }[] = JSON.parse(
    (await page.getByLabel('Details', { exact: true }).textContent())!,
  )
  const modes = reports.find((report) => report.thinking)?.thinking
  expect(modes).toHaveLength(2)
  const disabled = modes?.find((mode) => !mode.enabled)?.text
  const enabled = modes?.find((mode) => mode.enabled)?.text
  // WebLLM 0.2.86 exposes the empty, closed prefix even with thinking disabled.
  expect(disabled?.match(/<think>(.*?)<\/think>/s)?.[1]?.trim()).toBe('')
  expect(enabled?.match(/<think>(.*?)<\/think>/s)?.[1]?.trim()).toBeTruthy()
  expect(disabled).toContain('4')
  expect(enabled).toContain('4')
  await unload(page)
})

test('sampling is repeatable and reports real latency stages', async ({ page }) => {
  await load(page, 'sampling', 'main')
  await run(page)
  const details = (await page.getByLabel('Details', { exact: true }).textContent())!
  expect(details).toContain('"sameSeedMatches": true')
  expect(details).toContain('"stopped": "alpha"')
  for (const field of ['latencyBreakdown', 'penaltyTime', 'logitBiasTime', 'totalTime']) {
    expect(details).toContain(`"${field}"`)
  }
  const reports: { trace?: { latencyBreakdown?: Record<string, number[]> } }[] = JSON.parse(details)
  const timings = reports.flatMap((report) =>
    report.trace?.latencyBreakdown ? [report.trace.latencyBreakdown] : [],
  )
  expect(timings).toHaveLength(2)
  for (const timing of timings) {
    for (const field of ['penaltyTime', 'logitBiasTime', 'totalTime']) {
      const values = timing[field]!
      expect(values.length).toBeGreaterThan(0)
      expect(values.every((value) => Number.isFinite(value) && value >= 0)).toBe(true)
    }
    expect(timing.totalTime!.some((value) => value > 0)).toBe(true)
  }
  await unload(page)
})
