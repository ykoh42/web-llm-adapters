import { afterEach, expect, it, vi } from 'vitest'
import * as Tesseract from 'tesseract.js'
import { readFile, stat } from 'node:fs/promises'
import { imageText } from '../ocr.js'

vi.mock('tesseract.js', { spy: true })

afterEach(() => {
  vi.restoreAllMocks()
  vi.clearAllMocks()
})

it('starts no worker when the request is already cancelled', async () => {
  const create = vi.mocked(Tesseract.createWorker)
  const controller = new AbortController()
  controller.abort(new Error('Stop OCR'))
  await expect(imageText(new Uint8Array(), controller.signal)).rejects.toThrow('Stop OCR')
  expect(create).not.toHaveBeenCalled()
})

it.each(['initialization', 'recognition'])(
  'closes the worker after cancellation during %s',
  async (stage) => {
    const controller = new AbortController()
    const { createWorker: original } = await vi.importActual<typeof Tesseract>('tesseract.js')
    let worker: Tesseract.Worker | undefined
    vi.mocked(Tesseract.createWorker).mockImplementationOnce(async (...args) => {
      worker = await original(...args)
      vi.spyOn(worker, 'terminate')
      if (stage === 'initialization') controller.abort(new Error('Stop OCR'))
      else
        vi.spyOn(worker, 'recognize').mockImplementationOnce(() => {
          controller.abort(new Error('Stop OCR'))
          return new Promise(() => {})
        })
      return worker
    })
    await expect(imageText(new Uint8Array(), controller.signal)).rejects.toThrow('Stop OCR')
    expect(worker?.terminate).toHaveBeenCalledOnce()
  },
)

it.each(['text', 'blank'])(
  'closes the worker and removes language files after a %s image',
  async (name) => {
    const { createWorker: original } = await vi.importActual<typeof Tesseract>('tesseract.js')
    let worker: Tesseract.Worker | undefined
    let dir = ''
    vi.mocked(Tesseract.createWorker).mockImplementationOnce(async (...args) => {
      dir = args[2]?.langPath ?? ''
      worker = await original(...args)
      vi.spyOn(worker, 'terminate')
      return worker
    })
    const bytes = await readFile(new URL(`./fixtures/${name}.png`, import.meta.url))
    const result = imageText(bytes, new AbortController().signal)
    if (name === 'blank') await expect(result).rejects.toThrow('No readable text')
    else expect(await result).toContain('583721')
    expect(worker?.terminate).toHaveBeenCalledOnce()
    await expect(stat(dir)).rejects.toMatchObject({ code: 'ENOENT' })
  },
)
