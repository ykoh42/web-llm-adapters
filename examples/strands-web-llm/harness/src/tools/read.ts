import { basename, dirname, isAbsolute } from 'node:path'
import { read } from '@strands-agents/harness'
import { tool } from '@strands-agents/sdk'
import { extractText } from 'unpdf'
import { z } from 'zod'
import { imageText } from './ocr.ts'

const maxMediaBytes = 20 * 1024 * 1024

/** Extend Harness read with local PDF text extraction and image OCR. */
export function createRead() {
  return tool({
    name: 'read',
    description:
      'Read text files with line numbers, extract PDF text, or read English and Korean text in PNG/JPEG images with OCR. ' +
      'OCR extracts text, not objects or scenes. Audio, scanned PDFs and Office files are not supported. Offset and limit apply to text files.',
    inputSchema: z.object({
      path: z
        .string()
        .describe('Absolute path to the file.')
        .refine(
          (path) => isAbsolute(path) && !path.split(/[/\\]/).includes('..'),
          'Use an absolute path without .. segments.',
        ),
      offset: z.number().optional().describe('First text line, starting at 1.'),
      limit: z.number().optional().describe('Maximum number of text lines.'),
    }),
    callback: async (input, context) => {
      if (!context) throw new Error('Read requires a tool context.')
      context.cancelSignal.throwIfAborted()
      const format = input.path.split('.').at(-1)?.toLowerCase()
      const image = format === 'png' || format === 'jpg' || format === 'jpeg'
      if (format === 'wav') throw new Error('Audio transcription is not available in this example.')
      if (format !== 'pdf' && !image) {
        const result = await read.invoke(input, context)
        if (typeof result !== 'string')
          throw new Error('This Gemma build cannot view this media format.')
        return result
      }
      const files = await context.agent.sandbox.listFiles(dirname(input.path)).catch(() => [])
      context.cancelSignal.throwIfAborted()
      const file = files.find((entry) => entry.name === basename(input.path))
      if (file?.size !== undefined && file.size > maxMediaBytes)
        throw new Error('Media files must be at most 20 MiB.')
      const bytes = await context.agent.sandbox.readFile(input.path)
      if (bytes.length > maxMediaBytes) throw new Error('Media files must be at most 20 MiB.')
      context.cancelSignal.throwIfAborted()
      if (image) return imageText(bytes, context.cancelSignal)
      const { text } = await extractText(new Uint8Array(bytes), { mergePages: true })
      context.cancelSignal.throwIfAborted()
      if (!text.trim()) throw new Error('This PDF has no extractable text. Scanned pages need OCR.')
      return text
    },
  })
}
