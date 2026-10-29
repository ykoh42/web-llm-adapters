# Pi Agent Core and Pi Durable with WebLLM

Run the agent, tools, and model inference in the browser. Model inference uses a caller-owned WebLLM Worker; no model API key or inference server is needed.

From the repository root:

```sh
pnpm install
pnpm dev:pi
```

Open the printed `/pi/` URL. Choose an example using the link at the top of either page:

| Example       | URL            | Entry point                                      | State                            |
| ------------- | -------------- | ------------------------------------------------ | -------------------------------- |
| Pi Agent Core | `/pi/`         | [agent.ts](browser/src/agent.ts)                 | A new `Agent` per prompt         |
| Pi Durable    | `/pi/durable/` | [durable-agent.ts](browser/src/durable-agent.ts) | One saved conversation per model |

Both examples run a `lookup` tool that returns `ORCHID-42`, then feed the result back to WebLLM for the final answer. The pages stream the answer and show the transcript. Cancel interrupts generation and keeps the loaded model reusable.

The Agent Core example keeps its transcript for the current prompt. The Durable example saves one conversation per model in the browser. Reload the page and load the same model to restore it; an interrupted model request is generated again. Use one tab per conversation.

The Durable API is experimental. See the [official Durable guide](https://github.com/earendil-works/pi/blob/main/packages/durable/README.md) for details.

The first load downloads the model and runtime. Use a WebGPU-capable browser with sufficient GPU memory. Qwen3.5 4B is selected by default. The Gemma 4 option uses the runtime distributed with the adapter.

The example's `contextWindow` is 32768 and is passed to both the engine loader and the provider. Generation settings in [sampling.ts](sampling.ts) follow the model families; they are example settings rather than provider defaults.
