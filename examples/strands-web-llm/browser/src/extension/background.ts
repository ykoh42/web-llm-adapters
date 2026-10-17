import { ExtensionServiceWorkerMLCEngineHandler } from '@mlc-ai/web-llm'

let handler: ExtensionServiceWorkerMLCEngineHandler | undefined
chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== 'web_llm_service_worker') return
  if (handler) handler.setPort(port)
  else handler = new ExtensionServiceWorkerMLCEngineHandler(port)
  port.onMessage.addListener(handler.onmessage.bind(handler))
})
