import type { KnipConfig } from 'knip'

const config: KnipConfig = {
  workspaces: {
    'examples/harness': { entry: ['test/**/*.browser.ts'] },
  },
}
export default config
