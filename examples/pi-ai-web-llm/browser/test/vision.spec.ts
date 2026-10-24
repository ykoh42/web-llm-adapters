import { modelCheck } from '../../../../scripts/release/support.js'
import { test, expect } from './browser.js'
import type {} from './checks.js'

test(
  'Phi images match native WebLLM through Pi and recover after an invalid image',
  modelCheck('Phi-3.5-vision-instruct-q4f16_1-MLC', 'image'),
  async ({ page }) => {
    const errors: string[] = []
    page.on('pageerror', (error) => errors.push(error.message))
    await page.goto('http://127.0.0.1:4173/pi/test/checks.html')
    const result = await page.evaluate(() => window.piChecks.vision())

    expect(errors).toEqual([])
    expect(result.baseline.text.toLowerCase()).toContain('red')
    expect(result.failed.stopReason).toBe('error')
    expect(result.failed.errorMessage).toMatch(/decod/i)
    for (const [message, raw, native] of [
      [result.response, result.responseRaw, result.baseline],
      [result.answer, result.agentRaw, result.agentNative],
    ] as const) {
      expect(message.stopReason).toBe('stop')
      expect(
        message.content
          .filter((part) => part.type === 'text')
          .map((part) => part.text)
          .join(''),
      ).toBe(native.text)
      expect(raw).toBe(native.text)
      expect(message.usage.input).toBe(native.usage?.prompt_tokens)
      expect(message.usage.output).toBe(native.usage?.completion_tokens)
      expect(message.usage.totalTokens).toBe(native.usage?.total_tokens)
    }
    const user = result.agentRequest.messages.find((message) => message.role === 'user')
    expect(user?.content).toContainEqual({ type: 'image_url', image_url: { url: result.imageUrl } })
  },
)
