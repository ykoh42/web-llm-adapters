import { readFileSync } from 'node:fs'
import { checkReport } from './report.ts'
import { checkClean, git, reportPath, sourceHash, webllmVersion } from './source.ts'
import { supportClaims } from './support.ts'

checkClean()
const report = checkReport(
  JSON.parse(readFileSync(reportPath, 'utf8')),
  sourceHash(),
  webllmVersion(),
  supportClaims(readFileSync('README.md', 'utf8')),
)
git('merge-base', '--is-ancestor', report.commit, 'HEAD')
console.log(`Real models: ${report.tests.length} tests passed with WebLLM ${report.webllm}.`)
