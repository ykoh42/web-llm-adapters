import { createServer } from 'node:http'
import { readFile } from 'node:fs/promises'
import { extname, resolve, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('../../', import.meta.url))
const types: Record<string, string> = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.gif': 'image/gif',
  '.pdf': 'application/pdf',
  '.woff2': 'font/woff2',
  '.txt': 'text/plain',
  '.json': 'application/json',
  '.css': 'text/css',
  '.wasm': 'application/wasm',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
}

// One local origin lets real-model checks reuse WebLLM's existing browser cache.
createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url ?? '/', 'http://localhost').pathname)
    const prefix = ['/langchain/', '/pi/'].find((value) => pathname.startsWith(value))
    const adapter = prefix === '/pi/' ? 'pi-ai' : prefix === '/langchain/' ? 'langchain' : 'strands'
    const directory = resolve(root, `examples/${adapter}-web-llm/browser/dist`)
    const relative = pathname.slice(prefix?.length ?? 1)
    const file = resolve(
      directory,
      relative.endsWith('/') ? relative + 'index.html' : relative || 'index.html',
    )
    if (!file.startsWith(directory + sep)) {
      response.writeHead(403).end()
      return
    }
    const bytes = await readFile(file)
    response.writeHead(200, {
      'Content-Type': types[extname(file)] ?? 'application/octet-stream',
      'Content-Length': bytes.byteLength,
      'Service-Worker-Allowed': '/',
    })
    response.end(bytes)
  } catch {
    response.writeHead(404).end('Not found')
  }
}).listen(4173, '127.0.0.1', () => console.log('Local examples: http://127.0.0.1:4173/langchain/'))
