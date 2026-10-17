import { test, expect, load, run, unload, silentWav } from '../../fixtures/page.js'

for (const [engine, example] of [
  ['worker', 'chat'],
  ['service-worker', 'chat'],
  ['worker', 'tools'],
  ['service-worker', 'tools'],
] as const) {
  test(`Gemma ${example} streams through ${engine} and can stop then run again`, async ({
    page,
  }) => {
    await load(page, example, engine)
    await page.getByLabel('Message').fill('Count from 1 to 1000. Do not explain.')
    await page.getByRole('button', { name: 'Send', exact: true }).click()
    await expect(page.getByLabel('Response')).not.toBeEmpty({ timeout: 120_000 })
    await page.getByRole('button', { name: 'Stop', exact: true }).click()
    await expect(page.getByRole('status')).toHaveText('Stopped', { timeout: 30_000 })
    await page.getByLabel('Message').fill('Say hello.')
    await run(page)
    await unload(page)
  })
}

test('JSON follows its schema and keeps token logprobs', async ({ page }) => {
  await load(page, 'json')
  await run(page)
  expect(JSON.parse((await page.getByLabel('Response').textContent())!)).toEqual({
    name: 'apple',
    count: 3,
  })
  const details = await page.getByLabel('Details', { exact: true }).textContent()
  expect(details).toMatch(/"tokensWithLogprobs": [1-9]/)
  await unload(page)
})

test('grammar limits the answer to a color', async ({ page }) => {
  await load(page, 'grammar')
  await run(page)
  await expect(page.getByLabel('Response')).toHaveText('green')
  await unload(page)
})

test('native generation resumes after an interruption', async ({ page }) => {
  await load(page, 'resume')
  await run(page)
  await expect(page.getByLabel('Response')).toContainText('30')
  const details = await page.getByLabel('Details', { exact: true }).textContent()
  expect(details).toContain('"recoveredText"')
  expect(details).toMatch(/"recoveryMode": "(kv|token_replay)"/)
  await unload(page)
})

test('native text completion returns two choices', async ({ page }) => {
  await load(page, 'completions')
  await run(page)
  expect((await page.getByLabel('Response').textContent())!.split('\n\n---\n\n')).toHaveLength(2)
  await unload(page)
})

for (const example of ['audio', 'raw-audio']) {
  test(`${example} processes WAV input`, async ({ page }) => {
    await load(page, example)
    await page.getByLabel('Input file').setInputFiles(silentWav())
    await page.getByLabel('Message').fill('Describe any sound you hear in one sentence.')
    await run(page)
    await unload(page)
  })
}

test('native embeddings return a finite vector', async ({ page }) => {
  await load(page, 'embeddings')
  await run(page)
  const result = JSON.parse((await page.getByLabel('Response').textContent())!) as {
    dimensions: number
    firstValues: number[]
  }
  expect(result.dimensions).toBe(384)
  expect(result.firstValues).toHaveLength(8)
  expect(result.firstValues.every(Number.isFinite)).toBe(true)
  await unload(page)
})

for (const backend of ['cache', 'indexeddb', 'opfs']) {
  test(`${backend} stores and deletes a cached model`, async ({ page }) => {
    await page.getByLabel('Cache', { exact: true }).selectOption(backend)
    await load(page, 'embeddings')
    await run(page)
    await expect(
      page.getByRole('button', { name: 'Delete cached model', exact: true }),
    ).toBeDisabled()
    await unload(page)
    await page.getByRole('button', { name: 'Check cache', exact: true }).click()
    await expect(page.getByLabel('Details', { exact: true })).toContainText('"cached": true')
    await page.getByRole('button', { name: 'Delete cached model', exact: true }).click()
    await expect(page.getByRole('status')).toHaveText('Cached model removed')
    await page.getByRole('button', { name: 'Check cache', exact: true }).click()
    await expect(page.getByLabel('Details', { exact: true })).toContainText('"cached": false')
  })
}

test('Hermes calls a real Strands tool', async ({ page }) => {
  await load(page, 'hermes-tools')
  await page.getByRole('button', { name: 'Send', exact: true }).click()
  await expect(page.getByRole('status')).toHaveText('Done', { timeout: 120_000 })
  await expect(page.getByLabel('Response', { exact: true })).toBeEmpty()
  const details = await page.getByLabel('Details', { exact: true }).textContent()
  expect(details).toContain('"toolResultEvent"')
  expect(details).toMatch(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:/)
  await unload(page)
})

test('Gemma calls the clock and writes its own final answer', async ({ page }) => {
  await load(page, 'tools')
  await run(page)
  const details = await page.getByLabel('Details', { exact: true }).textContent()
  expect(details).toContain('"toolResultEvent"')
  const time = details!.match(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/)![0]
  // The answer must use the tool's time, not the fixed text from the legacy hook.
  await expect(page.getByLabel('Response')).toContainText(time.slice(11))
  await expect(page.getByLabel('Response')).not.toContainText('<|tool_call>')
  await unload(page)
})

test('Gemma thinking stays in Details while its clock answer stays in Response', async ({
  page,
}) => {
  await load(page, 'thinking-tools')
  await run(page)
  const details = await page.getByLabel('Details', { exact: true }).textContent()
  expect(details).toContain('"reasoning"')
  expect(details).toContain('"toolResultEvent"')
  const time = details!.match(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/)![0]
  await expect(page.getByLabel('Response')).toContainText(time.slice(11))
  await expect(page.getByLabel('Response')).not.toContainText('Reasoning:')
  await expect(page.getByLabel('Response')).not.toContainText(/<\|channel>|<channel\|>/)
  await unload(page)
})

test('Gemma can enable thinking without registering tools', async ({ page }) => {
  await load(page, 'gemma-thinking')
  await run(page)
  await expect(page.getByLabel('Response')).toContainText('4')
  await expect(page.getByLabel('Details', { exact: true })).toContainText('"reasoning"')
  await unload(page)
})

test('Phi vision describes a red image', async ({ page }) => {
  await load(page, 'image')
  const png = await page.evaluate(() => {
    const canvas = document.createElement('canvas')
    canvas.width = canvas.height = 256
    const context = canvas.getContext('2d')!
    context.fillStyle = 'red'
    context.fillRect(0, 0, 256, 256)
    return canvas.toDataURL('image/png').split(',')[1]!
  })
  await page.getByLabel('Input file').setInputFiles({
    name: 'red.png',
    mimeType: 'image/png',
    buffer: Buffer.from(png, 'base64'),
  })
  await run(page)
  await expect(page.getByLabel('Response')).toContainText(/red/i)
  await unload(page)
})
