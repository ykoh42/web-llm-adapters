import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

export const runtimeFiles = [
  'gemma-4-e2b.wasm',
  'manifest.json',
  'model.json',
  'cache.patch',
  'projection.patch',
  'README.md',
  'NOTICE.txt',
  'licenses/Apache-2.0.txt',
  'licenses/DLPack.txt',
  'licenses/Emscripten.txt',
  'licenses/LLVM.txt',
  'licenses/MLC-NOTICE.txt',
  'licenses/TVM-FFI-NOTICE.txt',
  'licenses/TVM-NOTICE.txt',
  'licenses/compiler-rt.txt',
  'licenses/musl.txt',
]

export function runtimeCopy() {
  return runtimeFiles.map((name) => ({
    from: fileURLToPath(new URL('runtime/' + name, import.meta.url)),
    to: join('dist/runtime', dirname(name)),
  }))
}
