import { stat, readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { stdout } from 'node:process'
import { startupJavaScriptFiles } from './startup-size.ts'

const root = resolve(import.meta.dirname, '..')
const limit = 40_000
const examples = [
  ['Pi AI', 'examples/pi-ai-web-llm/browser/dist'],
  ['Pi Durable', 'examples/pi-ai-web-llm/browser/dist', 'durable/index.html'],
  ['LangChain', 'examples/langchain-web-llm/browser/dist'],
  ['Strands', 'examples/strands-web-llm/browser/dist'],
  ['Strands Harness', 'examples/strands-web-llm/browser/dist', 'harness.html'],
]

for (const [name, directory, page = 'index.html'] of examples) {
  const dist = resolve(root, directory)
  const manifest = JSON.parse(await readFile(resolve(dist, '.vite/manifest.json'), 'utf8'))
  const files = startupJavaScriptFiles(manifest, page)
  if (!files.size) throw new Error(`${name} example has no startup JavaScript.`)

  let size = 0
  for (const file of files) size += (await stat(resolve(dist, file))).size
  if (size > limit) throw new Error(`${name} example startup JavaScript is ${size} bytes.`)

  stdout.write(`${name} example startup JavaScript: ${size} bytes.\n`)
}
