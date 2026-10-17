import { Offload, type ContextManagerConfig } from '@strands-agents/sdk'

/** Keep browser previews small; the SDK stash retains each full result. */
export function createContext(): ContextManagerConfig {
  return {
    strategies: [
      Offload.truncate('toolResults', { previewTokens: 256 }).when({ threshold: 2500 }),
      Offload.summarize('*').when({ utilization: 0.7, preserveRecent: 4 }),
    ],
  }
}
