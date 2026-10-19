import { defineConfig } from 'tsdown'
import { runtimeCopy } from '../../shared/build.ts'

export default defineConfig({
  entry: ['src/index.ts'],
  format: 'esm',
  fixedExtension: false,
  dts: true,
  clean: true,
  sourcemap: false,
  copy: runtimeCopy(),
})
