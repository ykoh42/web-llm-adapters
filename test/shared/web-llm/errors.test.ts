import { describe, expect, it } from 'vitest'
import { isContextOverflow } from '../../../shared/web-llm/errors.js'

describe('context overflow errors', () => {
  it.each(['ContextWindowSizeExceededError', 'EmbeddingExceedContextWindowSizeError'])(
    'recognizes a %s proxy without inspecting its prototype',
    (name) => {
      const error = new Proxy(Object.assign(new Error('Input is too long.'), { name }), {
        getPrototypeOf() {
          throw new Error('Cannot inspect provider prototype')
        },
      })

      expect(isContextOverflow(error)).toBe(true)
    },
  )
})
