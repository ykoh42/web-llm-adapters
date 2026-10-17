import { read } from '@strands-agents/harness'
import { tool } from '@strands-agents/sdk'
import { extractText } from 'unpdf'
import { z } from 'zod'

/** Extend Harness read with local PDF text extraction. */
export function createRead() {
  return tool({
    name: 'read',
    description:
      'Read text files with line numbers or extract PDF text. ' +
      'Images, audio, scanned PDFs and Office files are not supported. Offset and limit apply to text files.',
    inputSchema: z.object({
      path: z
        .string()
        .describe('Absolute path to the file.')
        .refine(
          (path) => path.startsWith('/') && !path.split(/[/\\]/).includes('..'),
          'Use an absolute path without .. segments.',
        ),
      offset: z.number().optional().describe('First text line, starting at 1.'),
      limit: z.number().optional().describe('Maximum number of text lines.'),
    }),
    callback: async (input, context) => {
      if (!context) throw new Error('Read requires a tool context.')
      context.cancelSignal.throwIfAborted()
      const format = input.path.split('.').at(-1)?.toLowerCase()
      if (format === 'wav') throw new Error('Audio transcription is not available in this example.')
      if (format !== 'pdf') {
        const result = await read.invoke(input, context)
        if (typeof result !== 'string')
          throw new Error('This Gemma build cannot view this media format.')
        return result
      }
      const bytes = await context.agent.sandbox.readFile(input.path)
      if (bytes.length > 20 * 1024 * 1024) throw new Error('PDF files must be at most 20 MiB.')
      context.cancelSignal.throwIfAborted()
      const { text } = await extractText(new Uint8Array(bytes), { mergePages: true })
      context.cancelSignal.throwIfAborted()
      if (!text.trim()) throw new Error('This PDF has no extractable text. Scanned pages need OCR.')
      return text
    },
  })
}
