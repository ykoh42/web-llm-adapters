import { inject } from 'vitest'

const error = inject('adapterBuildError')
if (error !== undefined) throw new Error(error)

declare module 'vitest' {
  export interface ProvidedContext {
    adapterBuildError: string | undefined
  }
}
