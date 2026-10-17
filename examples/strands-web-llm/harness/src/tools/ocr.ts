import { copyFile, mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createWorker, OEM, type Worker } from 'tesseract.js'

const files = [
  ['eng', import.meta.resolve('@tesseract.js-data/eng/4.0.0_best_int/eng.traineddata.gz')],
  ['kor', import.meta.resolve('@tesseract.js-data/kor/4.0.0_best_int/kor.traineddata.gz')],
] as const

/** Extract image text locally. Each call owns and closes its CPU worker. */
export async function imageText(bytes: Uint8Array, signal: AbortSignal) {
  signal.throwIfAborted()
  const dir = await mkdtemp(join(tmpdir(), 'webllm-ocr-'))
  let worker: Worker | undefined
  const cleanup = new AbortController()
  try {
    // Tesseract expects both language files in one local directory.
    await Promise.all(
      files.map(([code, path]) => copyFile(new URL(path), join(dir, code + '.traineddata.gz'))),
    )
    signal.throwIfAborted()
    worker = await createWorker('eng+kor', OEM.LSTM_ONLY, {
      langPath: dir,
      cacheMethod: 'none',
      // Job promises carry errors; avoid Tesseract's additional uncaught event throw.
      errorHandler: () => {},
    })
    signal.throwIfAborted()
    const cancelled = new Promise<never>((_, reject) => {
      signal.addEventListener('abort', () => reject(signal.reason), {
        once: true,
        signal: cleanup.signal,
      })
    })
    const { data } = await Promise.race([worker.recognize(Buffer.from(bytes)), cancelled])
    signal.throwIfAborted()
    if (!data.text.trim()) throw new Error('No readable text was found in this image.')
    return data.text
  } catch (error) {
    signal.throwIfAborted()
    throw error instanceof Error ? error : new Error(String(error))
  } finally {
    cleanup.abort()
    await Promise.all([worker?.terminate(), rm(dir, { recursive: true, force: true })])
  }
}
