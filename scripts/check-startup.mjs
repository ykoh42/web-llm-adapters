import { stat, readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { stdout } from 'node:process'

const root = resolve(import.meta.dirname, '..')
const limit = 40_000
const examples = [
  ['LangChain', 'examples/langchain-web-llm/browser/dist'],
  ['Strands', 'examples/strands-web-llm/browser/dist'],
]

for (const [name, directory] of examples) {
  const dist = resolve(root, directory)
  const html = await readFile(resolve(dist, 'index.html'), 'utf8')
  const files = new Set(
    Array.from(html.matchAll(/(?:src|href)="[^"]*\/assets\/([^"/]+\.js)"/g), ([, file]) => file),
  )
  if (!files.size) throw new Error(`${name} example has no startup JavaScript.`)

  let size = 0
  for (const file of files) size += (await stat(resolve(dist, 'assets', file))).size
  if (size > limit) throw new Error(`${name} example startup JavaScript is ${size} bytes.`)

  stdout.write(`${name} example startup JavaScript: ${size} bytes.\n`)
}
