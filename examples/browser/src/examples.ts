import { chat, grammar, json } from './chat.js'
import { tools } from './tools.js'
import { media } from './media.js'
import { completions, embeddings, rawAudio, resume } from './native.js'
import type { ExampleInput } from './output.js'
import { tags, thinking, sampling } from './advanced.js'

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
    id: 'tags',
    name: 'Structural tags',
    model: gemma,
    prompt: 'Return exactly <answer>{"count":3}</answer>. Do not explain.',
    run: tags,
  },
  {
    id: 'thinking',
    name: 'Qwen thinking on and off',
    model: 'Qwen3-0.6B-q4f16_1-MLC',
    prompt: 'What is 2 + 2? Answer briefly.',
    run: thinking,
  },
  {
    id: 'sampling',
    name: 'Sampling + latency',
    model: gemma,
    prompt: 'Give one short name for a friendly robot. Reply with only the name.',
    run: sampling,
  },
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
    name: 'Gemma tool call + answer',
    model: gemma,
    prompt: 'Use the clock tool to get the current UTC time, then tell me the time.',
    run: tools,
  },
  {
    id: 'gemma-thinking',
    name: 'Gemma thinking on and off',
    model: gemma,
    prompt: 'What is 2 + 2? Answer briefly.',
    contextWindow: 4096,
    run: thinking,
  },
  {
    id: 'thinking-tools',
    name: 'Gemma thinking + tools',
    model: gemma,
    prompt: 'Use the clock tool to get the current UTC time, then tell me the time.',
    contextWindow: 4096,
    run: (input) => tools(input, true),
  },
  {
    id: 'hermes-tools',
    name: 'Hermes tool batch',
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
