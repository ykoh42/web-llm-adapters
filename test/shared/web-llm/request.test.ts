import { expect, it } from 'vitest'
import {
  preserveThinkingRequest,
  streamOptions,
  type WebLLMRequest,
} from '../../../shared/web-llm/request.js'

it.each([
  { name: 'defaults to usage', request: {}, expected: { include_usage: true } },
  {
    name: 'keeps usage disabled',
    request: { stream_options: { include_usage: false } },
    expected: { include_usage: false },
  },
  { name: 'keeps null', request: { stream_options: null }, expected: null },
])('$name', ({ request, expected }) => {
  expect(streamOptions(request satisfies WebLLMRequest)).toEqual(expected)
})

it('keeps adapter options when a request replaces WebLLM extra body fields', () => {
  const defaults = {
    temperature: 0.5,
    extra_body: { enable_thinking: true, preserve_thinking: true },
  } satisfies WebLLMRequest
  const overrides = {
    temperature: 0.2,
    extra_body: { enable_latency_breakdown: true },
  } satisfies WebLLMRequest
  const request = structuredClone({ ...defaults, ...overrides })
  preserveThinkingRequest(request, overrides.extra_body, defaults.extra_body)

  expect(request).toEqual({
    temperature: 0.2,
    extra_body: { enable_latency_breakdown: true, preserve_thinking: true },
  })
  expect(defaults.extra_body).toEqual({ enable_thinking: true, preserve_thinking: true })
})

it('lets a higher-priority preserve-thinking option override the default', () => {
  const request = { extra_body: { enable_latency_breakdown: true } }
  preserveThinkingRequest(request, { preserve_thinking: false }, { preserve_thinking: true })

  expect(request.extra_body).toEqual({ enable_latency_breakdown: true, preserve_thinking: false })
})
