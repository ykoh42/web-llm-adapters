import type { KnipConfig } from 'knip'

const config: KnipConfig = {
  workspaces: {
    // Keep one SDK runtime across workspaces: Harness checks instanceof Model.
    '.': { ignoreDependencies: ['@strands-agents/sdk'] },
    'examples/browser': { entry: ['test/**/*.spec.ts'] },
    'examples/harness': { entry: ['test/**/*.browser.ts', 'test/install/*.ts'] },
  },
}
export default config
