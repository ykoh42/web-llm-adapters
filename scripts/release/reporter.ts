import { readFileSync, rmSync, writeFileSync } from 'node:fs'
import type { FullResult, Reporter, TestCase, TestResult } from '@playwright/test/reporter'
import { format, resolveConfig } from 'prettier'
import { checkReport, modelReport, type ModelReport } from './report.ts'
import { checkClean, git, reportPath, sourceHash, webllmVersion } from './source.ts'
import { supportClaims } from './support.ts'

export default class ModelReporter implements Reporter {
  private report?: ModelReport
  private error?: unknown

  onTestBegin() {
    if (this.report || this.error) return
    try {
      this.report = {
        commit: git('rev-parse', 'HEAD'),
        source: sourceHash(),
        webllm: webllmVersion(),
        checkedAt: new Date().toISOString(),
        environment: modelReport.shape.environment.parse(
          JSON.parse(process.env.WEB_LLM_RELEASE_ENVIRONMENT ?? 'null'),
        ),
        tests: [],
      }
    } catch (error) {
      this.error = error
    }
  }

  onTestEnd(test: TestCase, result: TestResult) {
    if (!this.report) return
    try {
      this.report.tests.push({
        adapter: modelReport.shape.tests.element.shape.adapter.parse(test.parent.project()?.name),
        name: test.titlePath().slice(2).join(' > '),
        status: result.status,
        checks: test.annotations
          .filter((annotation) => annotation.type === 'supports')
          .map((annotation) => annotation.description ?? ''),
      })
    } catch (error) {
      this.error = error
    }
  }

  async onEnd(result: FullResult) {
    // A reporter error must fail the run; Playwright otherwise ignores reporter exceptions.
    try {
      if (this.error) throw this.error
      if (result.status !== 'passed') throw new Error('Real-model run did not pass.')
      if (!this.report) return
      checkClean()
      checkReport(
        this.report,
        sourceHash(),
        webllmVersion(),
        supportClaims(readFileSync('README.md', 'utf8')),
      )
      writeFileSync(
        reportPath,
        await format(JSON.stringify(this.report), {
          ...(await resolveConfig(reportPath)),
          filepath: reportPath,
        }),
      )
      return { status: 'passed' as const }
    } catch (error) {
      try {
        rmSync(reportPath, { force: true })
      } catch (cleanupError) {
        console.error(cleanupError)
      }
      console.error(error)
      return { status: 'failed' as const }
    }
  }
}
