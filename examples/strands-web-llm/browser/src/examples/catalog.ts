import type { ExampleInput } from './output.js'

interface Example {
  id: string
  name: string
  model: string
  prompt: string
  run: (input: ExampleInput) => Promise<void>
  accept?: string
  contextWindow?: number
}

const gemma = 'gemma-4-E2B-it-q4f16_1-MLC'

export const examples: Example[] = [
  {
    id: 'chat',
    name: 'Chat',
    model: gemma,
    prompt: 'Reply with a short greeting.',
    run: async (input) => (await import('./chat.js')).chat(input),
  },
  {
    id: 'tags',
    name: 'Structural tags',
    model: gemma,
    prompt: 'Return exactly <answer>{"count":3}</answer>. Do not explain.',
    run: async (input) => (await import('./advanced.js')).tags(input),
  },
  {
    id: 'thinking',
    name: 'Qwen thinking on and off',
    model: 'Qwen3-0.6B-q4f16_1-MLC',
    prompt: 'What is 2 + 2? Answer briefly.',
    run: async (input) => (await import('./advanced.js')).thinking(input),
  },
  {
    id: 'sampling',
    name: 'Sampling + latency',
    model: gemma,
    prompt: 'Give one short name for a friendly robot. Reply with only the name.',
    run: async (input) => (await import('./advanced.js')).sampling(input),
  },
  {
    id: 'json',
    name: 'JSON schema + logprobs',
    model: gemma,
    prompt: 'Use name "apple" and count 3.',
    run: async (input) => (await import('./chat.js')).json(input),
  },
  {
    id: 'grammar',
    name: 'Grammar',
    model: gemma,
    prompt: 'Pick the color of grass: red, green, or blue.',
    run: async (input) => (await import('./chat.js')).grammar(input),
  },
  {
    id: 'tools',
    name: 'Gemma tool call + answer',
    model: gemma,
    prompt: 'Use the clock tool to get the current UTC time, then tell me the time.',
    run: async (input) => (await import('./tools.js')).tools(input),
  },
  {
    id: 'gemma-thinking',
    name: 'Gemma thinking on and off',
    model: gemma,
    prompt: 'What is 2 + 2? Answer briefly.',
    contextWindow: 4096,
    run: async (input) => (await import('./advanced.js')).thinking(input),
  },
  {
    id: 'thinking-tools',
    name: 'Gemma thinking + tools',
    model: gemma,
    prompt: 'Use the clock tool to get the current UTC time, then tell me the time.',
    contextWindow: 4096,
    run: async (input) => (await import('./tools.js')).tools(input, true),
  },
  {
    id: 'hermes-tools',
    name: 'Hermes tool batch',
    model: 'Hermes-3-Llama-3.1-8B-q4f16_1-MLC',
    prompt: 'Use the clock tool to get the current UTC time.',
    run: async (input) => (await import('./tools.js')).tools(input),
  },
  {
    id: 'image',
    name: 'Image',
    model: 'Phi-3.5-vision-instruct-q4f16_1-MLC',
    prompt: 'Describe this image in one sentence.',
    accept: 'image/png,image/jpeg',
    contextWindow: 8192,
    run: async (input) => (await import('./media.js')).media(input),
  },
  {
    id: 'audio',
    name: 'WAV audio',
    model: gemma,
    prompt: 'Transcribe this audio.',
    accept: '.wav',
    contextWindow: 8192,
    run: async (input) => (await import('./media.js')).media(input),
  },
  {
    id: 'raw-audio',
    name: 'PCM audio',
    model: gemma,
    prompt: 'Transcribe this audio.',
    accept: 'audio/*',
    contextWindow: 8192,
    run: async (input) => (await import('./media.js')).rawAudio(input),
  },
]
