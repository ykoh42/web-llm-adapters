import { chat, grammar, json } from './chat.js'
import { tools } from './tools.js'
import { media } from './media.js'
import { completions, embeddings, rawAudio, resume } from './native.js'
import type { ExampleInput } from './output.js'

interface Example {
  id: string
  name: string
  model: string
  prompt: string
  run: (input: ExampleInput) => Promise<void>
  accept?: string
  contextWindow?: number
  native?: boolean
}

const gemma = 'gemma-4-E2B-it-q4f16_1-MLC'

export const examples: Example[] = [
  { id: 'chat', name: 'Chat', model: gemma, prompt: 'Reply with a short greeting.', run: chat },
  {
    id: 'json',
    name: 'JSON schema + logprobs',
    model: gemma,
    prompt: 'Use name "apple" and count 3.',
    run: json,
  },
  {
    id: 'grammar',
    name: 'Grammar',
    model: gemma,
    prompt: 'Pick the color of grass: red, green, or blue.',
    run: grammar,
  },
  {
    id: 'tools',
    name: 'Tool call',
    model: 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC',
    prompt: 'Use the clock tool to get the current UTC time.',
    run: tools,
  },
  {
    id: 'image',
    name: 'Image',
    model: 'Phi-3.5-vision-instruct-q4f16_1-MLC',
    prompt: 'Describe this image in one sentence.',
    accept: 'image/png,image/jpeg',
    contextWindow: 8192,
    run: media,
  },
  {
    id: 'audio',
    name: 'WAV audio',
    model: gemma,
    prompt: 'Transcribe this audio.',
    accept: '.wav',
    contextWindow: 8192,
    run: media,
  },
  {
    id: 'raw-audio',
    name: 'Native PCM audio',
    model: gemma,
    prompt: 'Transcribe this audio.',
    accept: 'audio/*',
    contextWindow: 8192,
    native: true,
    run: rawAudio,
  },
  {
    id: 'embeddings',
    name: 'Native embeddings',
    model: 'snowflake-arctic-embed-s-q0f32-MLC-b4',
    contextWindow: 512,
    prompt: 'A small browser agent.',
    native: true,
    run: embeddings,
  },
  {
    id: 'completions',
    name: 'Native text completion',
    model: gemma,
    prompt: 'The sky is',
    native: true,
    run: completions,
  },
  {
    id: 'resume',
    name: 'Native save + resume',
    model: gemma,
    prompt: 'Count from 1 to 30, with spaces between numbers. Do not explain.',
    native: true,
    run: resume,
  },
]
