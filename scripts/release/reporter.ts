import { readFileSync, rmSync, writeFileSync } from 'node:fs'
import type { FullResult, Reporter, TestCase, TestResult } from '@playwright/test/reporter'
import { format, type FormatConfig } from 'oxfmt'
import { checkReport, checkTests, modelReport, type ModelReport } from './report.ts'
import { releaseTests, testName } from './tests.ts'
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
        name: testName(test.titlePath().slice(2)),
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
    // Setup invalidates the report before a run; a rejected preflight must leave it alone.
    if (result.status !== 'passed' && !this.report && !this.error)
      return { status: 'failed' as const }
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
      checkTests(this.report, releaseTests())
      const config: FormatConfig = JSON.parse(readFileSync('.oxfmtrc.json', 'utf8'))
      const { code, errors } = await format(reportPath, JSON.stringify(this.report), config)
      if (errors.length) throw new Error(errors.map((error) => error.message).join('\n'))
      writeFileSync(reportPath, code)
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
