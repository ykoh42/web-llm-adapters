/// <reference lib="webworker" />
import { ServiceWorkerMLCEngineHandler } from '@mlc-ai/web-llm'

declare const self: ServiceWorkerGlobalScope

// Register the handler during startup, including browser restarts of this worker.
new ServiceWorkerMLCEngineHandler()
self.addEventListener('install', (event) => event.waitUntil(self.skipWaiting()))
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()))
