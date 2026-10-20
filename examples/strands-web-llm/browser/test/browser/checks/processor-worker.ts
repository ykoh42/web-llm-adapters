import { WebWorkerMLCEngineHandler } from '@mlc-ai/web-llm'
import { FixedToken } from './fixed-token.js'
import { qwen } from './catalog.js'

const handler = new WebWorkerMLCEngineHandler()
handler.setLogitProcessorRegistry(new Map([[qwen, new FixedToken()]]))
self.onmessage = (event: MessageEvent) => handler.onmessage(event)
