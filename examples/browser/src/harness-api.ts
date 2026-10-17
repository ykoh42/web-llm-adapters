declare global {
  interface Window {
    harness: {
      open(modelId: string | string[], contextWindow?: number): Promise<void>
      close(): Promise<void>
    }
  }
}
export {}
