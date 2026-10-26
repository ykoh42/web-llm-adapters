import type { ChatCompletionContentPart } from '@mlc-ai/web-llm'
import { checkSampleRate, pcmWav } from '../../../../shared/web-llm/audio.js'
import { base64Data } from '../../../../shared/web-llm/base64.js'
import { isFloat32Array } from '../../../../shared/web-llm/typed-array.js'

export function formatPart(
  value: unknown,
  pcmFormat: 'pcm_f32' | 'wav' = 'pcm_f32',
): ChatCompletionContentPart {
  if (!isRecord(value)) throw new Error('WebLLM needs an object content block.')
  if (value.type === 'text' && typeof value.text === 'string')
    return { type: 'text', text: value.text }
  if (value.type === 'image_url') {
    const image = value.image_url
    const url = typeof image === 'string' ? image : isRecord(image) ? image.url : undefined
    const detail = isRecord(image) ? image.detail : undefined
    if (
      detail !== undefined &&
      detail !== null &&
      detail !== 'auto' &&
      detail !== 'low' &&
      detail !== 'high'
    )
      throw new Error('WebLLM needs image_url.detail to be auto, low, or high.')
    if (detail != null) throw new Error('WebLLM does not support image_url.detail.')
    if (typeof url === 'string') return { type: 'image_url', image_url: { url } }
  }
  const mime = value.mimeType ?? value.mime_type
  if (value.type === 'image') {
    if (typeof value.url === 'string') return { type: 'image_url', image_url: { url: value.url } }
    const data = base64Data(value.data)
    if (data !== undefined && typeof mime === 'string')
      return {
        type: 'image_url',
        image_url: { url: `data:${mime};base64,${data}` },
      }
    if (typeof value.base64 === 'string' && typeof value.mime_type === 'string')
      return {
        type: 'image_url',
        image_url: { url: `data:${value.mime_type};base64,${value.base64}` },
      }
  }
  if (value.type === 'input_audio' && isRecord(value.input_audio)) {
    if (value.input_audio.format === 'wav' && typeof value.input_audio.data === 'string')
      return { type: 'input_audio', input_audio: { format: 'wav', data: value.input_audio.data } }
    const { format, data, sample_rate } = value.input_audio
    if (format === 'pcm_f32' && isFloat32Array(data) && typeof sample_rate === 'number') {
      if (pcmFormat === 'wav') {
        // Keep native errors for detached or out-of-bounds views.
        const samples = data.byteLength ? data : new Float32Array(data)
        return {
          type: 'input_audio',
          input_audio: { format: 'wav', data: base64Data(pcmWav(samples, sample_rate))! },
        }
      }
      checkSampleRate(sample_rate)
      return {
        type: 'input_audio',
        input_audio: { format, data: new Float32Array(data), sample_rate },
      }
    }
  }
  if (value.type === 'audio') {
    let data: string | undefined
    if (typeof mime === 'string' && /^audio\/(?:wav|wave|x-wav)$/i.test(mime))
      data = base64Data(value.data)
    if (typeof value.url === 'string' && /^data:audio\/(?:wav|wave|x-wav);base64,/i.test(value.url))
      data = value.url
    if (data !== undefined) return { type: 'input_audio', input_audio: { format: 'wav', data } }
  }
  throw new Error(
    `WebLLM cannot use this ${String(value.type)} block. Use image URLs/base64, WAV, or Float32 PCM audio.`,
  )
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
