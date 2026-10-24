# Deep Agents

Start the [browser example](../README.md), then run from this directory:

```sh
pnpm start
```

The agent and default tools run in Node.js. WebLLM runs in its private browser page. `run.ts` owns that page and closes it when the agent finishes.

It writes and reads `/greeting.txt` in Deep Agents' virtual filesystem. The file is returned in `result.files`, not saved to your computer.

Requires Chrome with WebGPU. Model downloads are cached between runs.
