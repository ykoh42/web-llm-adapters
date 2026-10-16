import type { KnipConfig } from 'knip'

const config: KnipConfig = {
  workspaces: {
    'examples/browser': { entry: ['test/**/*.spec.ts'] },
    'examples/harness': { entry: ['test/**/*.browser.ts'] },
  },
}
export default config
