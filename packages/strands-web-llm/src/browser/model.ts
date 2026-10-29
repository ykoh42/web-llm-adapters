import {
  AudioBlock,
  ImageBlock,
  Message,
  ToolResultBlock,
  type ContentBlock,
  type ModelStreamEvent,
  type StreamOptions,
} from '@strands-agents/sdk'
import { fromBrowserError } from './errors.js'
import { callChunkCallback, waitForCallback } from '../../../../shared/web-llm/callback.js'
import { stringifyJSON } from '../../../../shared/web-llm/json.js'
import { checkBytes } from '../../../../shared/web-llm/base64.js'
import { startBrowserCall } from '../../../../shared/browser/start.js'
import { watchBrowserCall } from '../../../../shared/browser/lifecycle.js'
import { readBrowserError } from '../../../../shared/browser/errors.js'
import { WebLLMModelBase } from '../model-base.js'
import type { WebLLMModelConfig } from '../types.js'
import { formatSystemPrompt } from '../chat/messages.js'
import type { BrowserModelOptions, BrowserPage, BrowserRequest } from './types.js'

/** Runs the adapter in a controlled browser; Harness and its tools stay in Node.js. */
export class BrowserModel extends WebLLMModelBase {
  private readonly page: BrowserPage
  private readonly onChunk: BrowserModelOptions['onChunk']

  constructor(page: BrowserPage, { onChunk, ...config }: BrowserModelOptions) {
    super(config)
    this.page = page
    this.onChunk = onChunk
  }

  override async *stream(
    messages: Message[],
    options: StreamOptions = {},
  ): AsyncGenerator<ModelStreamEvent> {
    const { cancelSignal } = options
    cancelSignal?.throwIfAborted()
    const stopCall = (id: string) =>
      this.page.evaluate((id) => {
        const bridge = window.webllmBridge
        if (!bridge) throw new Error('The WebLLM browser bridge is no longer available.')
        return bridge.stop(id)
      }, id)
    const start = this.page.evaluate(
      (input) => {
        const request: BrowserRequest = JSON.parse(input)
        const bridge = window.webllmBridge
        if (!bridge)
          throw new Error(
            'Expose createBrowserBridge(engine) as window.webllmBridge before using BrowserModel.',
          )
        return bridge.start(request)
      },
      encodeRequest(messages, this.getConfig(), options, Boolean(this.onChunk)),
    )
    const id = await startBrowserCall(start, stopCall, cancelSignal)
    const finish = watchBrowserCall(id, stopCall, cancelSignal)
    let failed = false
    let complete = false
    try {
      cancelSignal?.throwIfAborted()
      while (true) {
        const result = await waitForCallback(this.page.evaluate(readBrowserEvent, id), cancelSignal)
        for (const chunk of result.chunks ?? []) {
          cancelSignal?.throwIfAborted()
          const pending = callChunkCallback(this.onChunk, chunk, cancelSignal)
          if (pending) await pending
        }
        cancelSignal?.throwIfAborted()
        if ('error' in result) {
          complete = true
          throw fromBrowserError(readBrowserError(result))
        }
        if (result.done) {
          complete = true
          return
        }
        if ('native' in result) continue
        yield result.value
      }
    } catch (error) {
      failed = true
      throw error
    } finally {
      const pending = finish(complete, failed)
      if (pending) await pending
    }
  }
}

function readBrowserEvent(id: string) {
  const bridge = window.webllmBridge
  if (!bridge) throw new Error('The WebLLM browser bridge is no longer available.')
  return bridge.next(id)
}

/** Send only provider inputs; SDK message JSON preserves media bytes as base64. */
function encodeRequest(
  messages: Message[],
  config: WebLLMModelConfig,
  options: StreamOptions,
  includeChunks = false,
) {
  const { systemPrompt, toolSpecs, toolChoice, dynamicTrailingBlocks } = options
  const prompt = formatSystemPrompt(systemPrompt)
  const request: BrowserRequest = {
    ...(includeChunks && { includeChunks }),
    messages: messages.map((message) => {
      const standard = message.toJSON === Message.prototype.toJSON
      const foreign = standard && checkMediaBytes(message.content)
      if (foreign) {
        const copy = new Message({
          role: message.role,
          content: message.content.map(normalizeBlock),
        })
        const { role, content } = copy.toJSON()
        return { role, content }
      }
      const { role, content } = message.toJSON()
      return { role, content }
    }),
    config,
    options: {
      ...(prompt !== undefined && { systemPrompt: prompt }),
      ...(toolSpecs !== undefined && { toolSpecs }),
      ...(toolChoice !== undefined && { toolChoice }),
      ...(dynamicTrailingBlocks !== undefined && { dynamicTrailingBlocks }),
    },
  }
  // Browser transports may discard reserved object keys; JSON preserves their literal values.
  return stringifyJSON(request)
}

function checkMediaBytes(content: readonly ContentBlock[]): boolean {
  let foreign = false
  const check = (block: ContentBlock | ToolResultBlock['content'][number]) => {
    const bytes = mediaBytes(block)
    if (!bytes) return
    checkBytes(bytes)
    if (!(bytes instanceof Uint8Array)) foreign = true
  }
  for (const block of content) {
    if (block.type === 'toolResultBlock' && block.toJSON === ToolResultBlock.prototype.toJSON) {
      for (const part of block.content) check(part)
      continue
    }
    check(block)
  }
  return foreign
}

function normalizeBlock(block: ContentBlock): ContentBlock {
  if (block.type === 'toolResultBlock' && block.toJSON === ToolResultBlock.prototype.toJSON) {
    const content = block.content.map((part) => {
      const bytes = mediaBytes(part)
      if (!bytes || bytes instanceof Uint8Array || part.type !== 'imageBlock') return part
      return new ImageBlock({ ...part, source: { bytes: localBytes(bytes) } })
    })
    return new ToolResultBlock({ ...block, content })
  }
  const bytes = mediaBytes(block)
  if (!bytes || bytes instanceof Uint8Array) return block
  if (block.type === 'imageBlock')
    return new ImageBlock({ ...block, source: { bytes: localBytes(bytes) } })
  if (block.type === 'audioBlock')
    return new AudioBlock({ ...block, source: { bytes: localBytes(bytes) } })
  return block
}

function localBytes(bytes: Uint8Array): Uint8Array {
  return new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength)
}

function mediaBytes(
  block: ContentBlock | ToolResultBlock['content'][number],
): Uint8Array | undefined {
  if (
    block.type === 'imageBlock' &&
    block.toJSON === ImageBlock.prototype.toJSON &&
    block.source.type === 'imageSourceBytes'
  )
    return block.source.bytes
  if (
    block.type === 'audioBlock' &&
    block.toJSON === AudioBlock.prototype.toJSON &&
    block.source.type === 'audioSourceBytes'
  )
    return block.source.bytes
  return undefined
}
