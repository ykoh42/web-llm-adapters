import { execFileSync } from 'node:child_process'

interface TestList {
  suites: Suite[]
}

interface Suite {
  title: string
  suites?: Suite[]
  specs?: {
    title: string
    tests: { projectName: string }[]
  }[]
}

/** Collect every release test without launching a browser or running setup. */
export function releaseTests(): string[] {
  const output = execFileSync(
    process.execPath,
    [
      'node_modules/@playwright/test/cli.js',
      'test',
      '--config',
      'playwright.release.config.ts',
      '--list',
      '--reporter=json',
    ],
    {
      encoding: 'utf8',
      env: {
        ...process.env,
        MODEL_IDS: undefined,
        PLAYWRIGHT_JSON_OUTPUT_FILE: undefined,
        PLAYWRIGHT_JSON_OUTPUT_NAME: undefined,
        PLAYWRIGHT_JSON_OUTPUT_DIR: undefined,
      },
    },
  )
  const list: TestList = JSON.parse(output)
  const tests: string[] = []
  function visit(suite: Suite, parents: string[]) {
    const path = [...parents, suite.title]
    for (const spec of suite.specs ?? [])
      for (const test of spec.tests)
        tests.push(test.projectName + '/' + testName([...path, spec.title]))
    for (const child of suite.suites ?? []) visit(child, path)
  }
  for (const suite of list.suites) visit(suite, [])
  return tests
}

/** Playwright uses platform-specific separators for the test file title. */
export function testName([file, ...titles]: string[]): string {
  return [file!.replaceAll('\\', '/'), ...titles].join(' > ')
}
