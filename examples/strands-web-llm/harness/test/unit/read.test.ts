import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Agent, type ToolContext } from '@strands-agents/sdk'
import type { Page } from '@playwright/test'
import { BrowserModel } from 'strands-web-llm'
import { createRead } from '../../src/tools/read.js'
import { pdf } from '../../fixtures/pdf.js'

let dir: string
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'webllm-read-'))
})
afterEach(async () => {
  vi.restoreAllMocks()
  await rm(dir, { recursive: true, force: true })
})

function setup(signal = new AbortController().signal) {
  const model = new BrowserModel({} as Page, { modelId: 'gemma-4-E2B-it-q4f16_1-MLC' })
  const agent = new Agent({ model, printer: false })
  const context: ToolContext = {
    agent,
    toolUse: { name: 'read', toolUseId: 'read-1', input: {} },
    invocationState: {},
    interrupt: vi.fn(),
    cancelSignal: signal,
  }
  return { context, tool: createRead() }
}

it('keeps the built-in text reader and its paging behavior', async () => {
  const { tool, context } = setup()
  const path = join(dir, 'note.txt')
  await writeFile(path, 'first\nsecond\nthird')
  const text = await tool.invoke({ path, offset: 2, limit: 1 }, context)
  expect(text).toContain('     2\tsecond')
  expect(text).not.toContain('first')
  expect(text).not.toContain('third')
})

it('extracts real PDF text without calling a model', async () => {
  const { tool, context } = setup()
  const path = join(dir, 'note.PDF')
  const code = 'pdf-' + crypto.randomUUID()
  await writeFile(path, pdf(`Value: ${code}`))
  expect(await tool.invoke({ path }, context)).toContain(code)
})

it('rejects cancelled work before reading a file', async () => {
  const controller = new AbortController()
  controller.abort(new Error('Stop reading'))
  const { tool, context } = setup(controller.signal)
  const read = vi.spyOn(context.agent.sandbox, 'readFile')
  const list = vi.spyOn(context.agent.sandbox, 'listFiles')
  await expect(tool.invoke({ path: join(dir, 'note.pdf') }, context)).rejects.toThrow(
    'Stop reading',
  )
  expect(read).not.toHaveBeenCalled()
  expect(list).not.toHaveBeenCalled()
})

it('stops after cancellation during the size check', async () => {
  const controller = new AbortController()
  const { tool, context } = setup(controller.signal)
  vi.spyOn(context.agent.sandbox, 'listFiles').mockImplementation(async () => {
    controller.abort(new Error('Stop reading'))
    return []
  })
  const read = vi.spyOn(context.agent.sandbox, 'readFile')

  await expect(tool.invoke({ path: '/sandbox/note.pdf' }, context)).rejects.toThrow('Stop reading')
  expect(read).not.toHaveBeenCalled()
})

it.each(['relative.pdf', '/tmp/../note.pdf'])('rejects invalid path %s', async (path) => {
  const { tool, context } = setup()
  const read = vi.spyOn(context.agent.sandbox, 'readFile')
  await expect(tool.invoke({ path }, context)).rejects.toThrow()
  expect(read).not.toHaveBeenCalled()
})

it.each(['pdf', 'png'])('rejects oversized %s files', async (format) => {
  const { tool, context } = setup()
  vi.spyOn(context.agent.sandbox, 'readFile').mockResolvedValueOnce(
    new Uint8Array(20 * 1024 * 1024 + 1),
  )
  await expect(tool.invoke({ path: join(dir, 'large.' + format) }, context)).rejects.toThrow(
    '20 MiB',
  )
})

it.each(['pdf', 'png'])(
  'rejects oversized %s files before reading when size is known',
  async (format) => {
    const { tool, context } = setup()
    const path = join(dir, 'large.' + format)
    vi.spyOn(context.agent.sandbox, 'listFiles').mockResolvedValueOnce([
      { name: 'large.' + format, size: 20 * 1024 * 1024 + 1 },
    ])
    const read = vi.spyOn(context.agent.sandbox, 'readFile')

    await expect(tool.invoke({ path }, context)).rejects.toThrow('20 MiB')
    expect(read).not.toHaveBeenCalled()
  },
)

it.each(['gif', 'docx', 'wav'])('rejects unsupported %s media', async (format) => {
  const { tool, context } = setup()
  const path = join(dir, 'note.' + format)
  const bytes =
    format === 'gif'
      ? Buffer.from('R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7', 'base64')
      : new Uint8Array([1, 2, 3])
  await writeFile(path, bytes)
  await expect(tool.invoke({ path }, context)).rejects.toThrow(/media format|Audio transcription/)
})

it.each(['png', 'jpeg'])('reads English and Korean text from a real %s image', async (format) => {
  const { tool, context } = setup()
  const bytes = await readFile(new URL(`../../fixtures/data/text.${format}`, import.meta.url))
  // Use sandbox bytes rather than a host file; OCR must preserve that boundary.
  const read = vi.spyOn(context.agent.sandbox, 'readFile').mockResolvedValueOnce(bytes)
  const text = await tool.invoke({ path: '/sandbox/note.' + format }, context)
  expect(read).toHaveBeenCalledWith('/sandbox/note.' + format)
  expect(text).toContain('Verification code: 583721')
  expect(text).toContain('확인 번호: 583721')
})

it('rejects images with no readable text', async () => {
  const { tool, context } = setup()
  const bytes = await readFile(new URL('../../fixtures/data/blank.png', import.meta.url))
  vi.spyOn(context.agent.sandbox, 'readFile').mockResolvedValueOnce(bytes)
  await expect(tool.invoke({ path: '/sandbox/blank.png' }, context)).rejects.toThrow(
    'No readable text',
  )
})

it('rejects malformed images before starting OCR', async () => {
  const { tool, context } = setup()
  vi.spyOn(context.agent.sandbox, 'readFile').mockResolvedValueOnce(new Uint8Array([1, 2, 3]))
  await expect(tool.invoke({ path: '/sandbox/bad.png' }, context)).rejects.toThrow(
    'PNG or JPEG data',
  )
})

it('rejects PDFs with no extractable text', async () => {
  const { tool, context } = setup()
  const path = join(dir, 'blank.pdf')
  await writeFile(path, pdf(''))
  await expect(tool.invoke({ path }, context)).rejects.toThrow('no extractable text')
})

it('requires a tool context', async () => {
  await expect(createRead().invoke({ path: '/tmp/note.pdf' })).rejects.toThrow('tool context')
})
