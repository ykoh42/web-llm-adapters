# Pi Agent Core and Pi Durable with WebLLM

Run the agent, tools, and model inference in the browser. Model inference uses a caller-owned WebLLM Worker; no model API key or inference server is needed.

From the repository root:

```sh
npm install
npm run dev:pi
```

Open the printed `/pi/` URL. Choose an example using the link at the top of either page:

| Example       | URL            | Entry point                                      | State                            |
| ------------- | -------------- | ------------------------------------------------ | -------------------------------- |
| Pi Agent Core | `/pi/`         | [agent.ts](browser/src/agent.ts)                 | A new `Agent` per prompt         |
| Pi Durable    | `/pi/durable/` | [durable-agent.ts](browser/src/durable-agent.ts) | One saved conversation per model |

Both examples run a `lookup` tool that returns `ORCHID-42`, then feed the result back to WebLLM for the final answer. The pages stream the answer and show the transcript. Cancel interrupts generation and keeps the loaded model reusable.

The Agent Core example passes `models.streamSimple.bind(models)` to `Agent`. Its transcript lives only for the current prompt.

The Durable example passes the same model collection to `Harness.open()`, registers the tool through `defineExtension()` and `defineTool()`, and submits inputs to the root conversation. Reload the page and load the same model to restore the transcript and resume unfinished work from its last committed checkpoint. An interrupted local model request is generated again; the GPU state itself is not saved.

[storage.ts](browser/src/storage.ts) implements the public Durable `Storage` contract using `MemoryStorage.prepareCommit()` and an IndexedDB commit log. Each commit is persisted before it is published. A Web Lock allows one tab to own a conversation database at a time. This small example retains its entire log; it does not implement compaction or a storage quota policy. Browser storage is local to the current origin and browser profile and can be cleared or evicted by the browser. Clear the `pi-webllm-*` IndexedDB databases in browser developer tools to remove saved conversations.

The Durable API is experimental; this example pins `@earendil-works/pi-durable` and `@earendil-works/chord` to 1.3.0. See the [official Durable guide](https://github.com/earendil-works/pi/blob/main/packages/durable/README.md) for the Harness and storage contracts.

The first load downloads the model and runtime. Use a WebGPU-capable browser with sufficient GPU memory. Qwen3.5 4B is selected by default. The Gemma 4 option uses the runtime distributed with the adapter.

The example's `contextWindow` is 32768 and is passed to both the engine loader and the provider. Generation settings in [sampling.ts](sampling.ts) follow the model families; they are example settings rather than provider defaults.

[engine.ts](browser/src/engine.ts) owns model loading and provider registration. [main.ts](browser/src/main.ts) connects Agent Core to its page; [durable.ts](browser/src/durable.ts) connects the Durable conversation to its page. Browser diagnostics live under `browser/test/` and run on a separate test page.

```sh
npm run test:pi
```

The real-model checks use installed Chrome and the repository's `.cache/chrome` test profile and retain transcripts under `test-results/pi-ai-web-llm`. They verify tools and answers on four exact model artifacts in both Agent Core and Durable, cancellation and recovery in both, Durable reload and follow-up, interrupted-turn recovery, and the single-writer lock. Set `HEADLESS=1` to request headless Chrome where hardware WebGPU is available. Do not run other tests using the same persistent profile simultaneously.
