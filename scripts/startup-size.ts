interface ManifestChunk {
  file?: string
  imports?: string[]
}

type Manifest = Record<string, ManifestChunk>

export function startupJavaScriptFiles(manifest: Manifest, entry: string) {
  const files = new Set<string>()
  const visited = new Set<string>()

  function visit(key: string) {
    if (visited.has(key)) return
    visited.add(key)

    const chunk = manifest[key]
    if (!chunk) throw new Error(`Startup import is missing from the Vite manifest: ${key}`)
    if (chunk.file?.endsWith('.js')) files.add(chunk.file)

    for (const imported of chunk.imports ?? []) visit(imported)
  }

  visit(entry)
  return files
}
