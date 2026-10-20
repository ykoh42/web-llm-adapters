import type { LogitProcessor } from '@mlc-ai/web-llm'

export class FixedToken implements LogitProcessor {
  readonly sampled: number[] = []
  resets = 0

  processLogits(logits: Float32Array): Float32Array {
    logits.fill(-Infinity)
    logits[42] = 0
    return logits
  }

  processSampledToken(token: number): void {
    this.sampled.push(token)
  }

  resetState(): void {
    this.sampled.length = 0
    this.resets++
  }
}
