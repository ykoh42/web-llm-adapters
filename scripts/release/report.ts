import { z } from 'zod'

export const modelReport = z.object({
  commit: z.string().regex(/^[a-f0-9]{40}$/),
  source: z.string().regex(/^[a-f0-9]{64}$/),
  webllm: z.string().min(1),
  checkedAt: z.iso.datetime(),
  environment: z.object({
    platform: z.string(),
    arch: z.string(),
    node: z.string(),
    browser: z.string().min(1),
    gpu: z.object({
      vendor: z.string(),
      architecture: z.string(),
      device: z.string(),
      description: z.string(),
    }),
  }),
  tests: z
    .array(
      z.object({
        adapter: z.enum(['langchain', 'pi', 'strands']),
        name: z.string().min(1),
        status: z.enum(['passed', 'failed', 'timedOut', 'skipped', 'interrupted']),
        checks: z.array(z.string().min(1)),
      }),
    )
    .min(1),
})
export type ModelReport = z.infer<typeof modelReport>

export function checkReport(value: unknown, source: string, webllm: string, claims: string[]) {
  const report = modelReport.parse(value)
  if (report.source !== source || report.webllm !== webllm)
    throw new Error('Real-model results are stale. Run pnpm test:release:models.')
  if (report.tests.some((test) => test.status !== 'passed'))
    throw new Error('Every release model test must pass without skips or retries.')
  const checks = new Set(
    report.tests.flatMap((test) => test.checks.map((check) => `${test.adapter}/${check}`)),
  )
  const missing = claims.filter((claim) => !checks.has(claim))
  if (missing.length) throw new Error(`Missing real-model checks:\n${missing.join('\n')}`)
  return report
}

/** Require every release test, including tests that do not add a support-table claim. */
export function checkTests(report: ModelReport, expected: readonly string[]) {
  const actual = report.tests.map((test) => `${test.adapter}/${test.name}`)
  const names = new Set(actual)
  const missing = expected.filter((name) => !names.has(name))
  if (missing.length || actual.length !== expected.length || names.size !== actual.length)
    throw new Error(
      `Real-model results must include every release test. Missing:\n${missing.join('\n')}`,
    )
}
