import type { KnipConfig } from 'knip'

const config: KnipConfig = {
  workspaces: {
    'examples/strands-web-llm/browser': { entry: ['test/**/*.spec.ts'] },
    'examples/strands-web-llm/harness': {
      entry: ['test/browser/*.browser.ts', 'test/install/*.ts'],
    },
  },
}
export default config
