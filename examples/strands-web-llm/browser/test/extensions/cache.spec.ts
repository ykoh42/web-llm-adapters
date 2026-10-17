import { test, expect } from '../../fixtures/extension.js'
import { run, unload } from '../../fixtures/page.js'

// Ports create different origins; only the real extension can share these model weights.
test('two origins share model weights without fetching them again', async ({ page, context }) => {
  for (const port of [4173, 4174]) {
    await page.goto(`http://localhost:${port}`)
    await page.waitForFunction(() => 'crossOriginStorage' in navigator)
    await page.getByLabel('Cache', { exact: true }).selectOption('cross-origin')
    await page.getByLabel('Engine', { exact: true }).selectOption('main')
    await page.getByLabel('Model', { exact: true }).selectOption('Qwen3-0.6B-q4f16_1-MLC')
    await expect(
      page.getByRole('button', { name: 'Delete cached model', exact: true }),
    ).toBeDisabled()
    const blocked: string[] = []
    if (port === 4174) {
      await context.route('**/*', async (route) => {
        const url = route.request().url()
        // /raw/ supplies the small LFS hash pointer; /resolve/ downloads the weights.
        if (url.includes('/resolve/') && /params_shard.*\.bin/.test(url)) {
          blocked.push(url)
          console.log('Blocked model download:', url)
          await route.abort()
        } else await route.continue()
      })
    }
    await page.getByRole('button', { name: 'Load model', exact: true }).click()
    await page.waitForFunction(
      () =>
        !document.querySelector<HTMLButtonElement>('#load')!.disabled ||
        !document.querySelector<HTMLButtonElement>('#run')!.disabled,
      undefined,
      { timeout: 12 * 60 * 1000 },
    )
    await expect(page.getByRole('status')).toHaveText('Model loaded')
    await run(page)
    expect(blocked).toEqual([])
    await unload(page)
  }
})
