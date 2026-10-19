# Deep Agents

Start the [browser example](../README.md), then run from this directory:

```sh
npm start
```

The agent and default tools run in Node.js. WebLLM runs in its private browser page. `run.ts` owns that page and closes it when the agent finishes.

Requires Chrome with WebGPU. Model downloads are cached between runs.
