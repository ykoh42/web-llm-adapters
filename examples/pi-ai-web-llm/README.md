# Pi AI and Pi Agent Core with WebLLM

Run the agent, tools, and model inference in the browser. Model inference uses a caller-owned WebLLM Worker; no model API key or inference server is needed.

From the repository root:

```sh
npm install
npm run dev:pi
```

Open the printed `/pi/` URL, load a model, then ask it to look up `A7`. The `lookup` tool returns `ORCHID-42`; Pi Agent Core feeds the result back to WebLLM for the final answer. The page streams the answer and shows the transcript. Cancel interrupts generation and keeps the loaded model reusable.

The first load downloads the model and runtime. Use a WebGPU-capable browser with sufficient GPU memory. Qwen3.5 4B is selected by default. The Gemma 4 option uses the runtime distributed with the adapter.

The example's `contextWindow` is 32768 and is passed to both the engine loader and the provider. Generation settings in [sampling.ts](sampling.ts) follow the model families; they are example settings rather than provider defaults.

[engine.ts](browser/src/engine.ts) owns model loading and provider registration. [agent.ts](browser/src/agent.ts) creates the Pi agent and its tool. [main.ts](browser/src/main.ts) connects them to the page. Browser diagnostics live under `browser/test/` and run on a separate test page.

```sh
npm run test:pi
```

The real-model checks use installed Chrome and the repository's `.cache/chrome` test profile, verify four exact model artifacts, and retain transcripts under `test-results/pi-ai-web-llm`. Set `HEADLESS=1` to request headless Chrome where hardware WebGPU is available. Do not run other tests using the same persistent profile simultaneously.
