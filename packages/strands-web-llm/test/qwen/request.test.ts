import { Message, TextBlock, ToolResultBlock, ToolUseBlock } from '@strands-agents/sdk'
import { expect, it } from 'vitest'
import { qwenRequest } from '../../src/qwen/request.js'

it.each(['Qwen2.5-3B-Instruct-q4f16_1-MLC', 'Qwen3-4B-q4f16_1-MLC'])(
  '%s keeps the official line breaks around tool history',
  (modelId) => {
    const input = { path: '/tmp/note.txt', offset: 1, limit: 1 }
    const request = qwenRequest(
      { modelId },
      [
        new Message({
          role: 'assistant',
          content: [new ToolUseBlock({ toolUseId: 'read-1', name: 'read', input })],
        }),
        new Message({
          role: 'user',
          content: [
            new ToolResultBlock({
              toolUseId: 'read-1',
              status: 'success',
              content: [new TextBlock('The code is oak-42.')],
            }),
          ],
        }),
      ],
      {},
    )
    const assistant = request.messages[0]!.content as string
    const lines = assistant.split('\n')
    expect(lines[0]).toBe('<tool_call>')
    expect(lines.at(-1)).toBe('</tool_call>')
    expect(JSON.parse(lines.slice(1, -1).join('\n'))).toEqual({ name: 'read', arguments: input })
    expect(request.messages[1]).toEqual({
      role: 'user',
      content: '<tool_response>\nThe code is oak-42.\n</tool_response>',
    })
  },
)
