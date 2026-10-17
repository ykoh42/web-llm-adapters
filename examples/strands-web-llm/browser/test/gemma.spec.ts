import { test, expect, load, unload } from './helpers.js'

test('Gemma 4 E2B streams through a Strands Agent', async ({ page }) => {
  await load(page, 'chat', 'main')
  const output = page.getByLabel('Response')
  await output.evaluate((element) => {
    let updates = 0
    new MutationObserver(() => {
      const status = document.querySelector('[role="status"]')?.textContent
      if (element.textContent && status === 'Running…') {
        element.setAttribute('data-updates', String(++updates))
      }
    }).observe(element, { childList: true, characterData: true, subtree: true })
  })
  await page
    .getByLabel('Message')
    .fill('Count from 1 to 20. Separate numbers with spaces. Do not explain.')
  await page.getByRole('button', { name: 'Send', exact: true }).click()
  await expect(page.getByRole('status')).toHaveText('Done', { timeout: 120_000 })
  await expect(output).not.toBeEmpty()
  expect(Number(await output.getAttribute('data-updates'))).toBeGreaterThan(1)
  console.log('Gemma response:', await output.textContent())
  console.log('Visible stream updates:', await output.getAttribute('data-updates'))
  await unload(page)
})
