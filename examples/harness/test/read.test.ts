import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Agent, type ToolContext } from '@strands-agents/sdk'
import type { Page } from '@playwright/test'
import { BrowserModel } from '../browser-model.js'
import { createRead } from '../read.js'
import { pdf } from './pdf.js'

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
  await expect(tool.invoke({ path: join(dir, 'note.pdf') }, context)).rejects.toThrow(
    'Stop reading',
  )
  expect(read).not.toHaveBeenCalled()
})

it.each(['relative.pdf', '/tmp/../note.pdf'])('rejects invalid path %s', async (path) => {
  const { tool, context } = setup()
  const read = vi.spyOn(context.agent.sandbox, 'readFile')
  await expect(tool.invoke({ path }, context)).rejects.toThrow()
  expect(read).not.toHaveBeenCalled()
})

it('rejects oversized PDFs', async () => {
  const { tool, context } = setup()
  vi.spyOn(context.agent.sandbox, 'readFile').mockResolvedValueOnce(
    new Uint8Array(20 * 1024 * 1024 + 1),
  )
  await expect(tool.invoke({ path: join(dir, 'large.pdf') }, context)).rejects.toThrow('20 MiB')
})

it.each(['png', 'docx', 'wav'])('rejects unsupported %s media', async (format) => {
  const { tool, context } = setup()
  const path = join(dir, 'note.' + format)
  await writeFile(path, new Uint8Array([1, 2, 3]))
  await expect(tool.invoke({ path }, context)).rejects.toThrow(/media format|Audio transcription/)
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
