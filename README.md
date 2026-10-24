# WebLLM Adapters

[WebLLM](https://github.com/mlc-ai/web-llm) runs AI models locally in your browser using WebGPU.
These community adapters connect WebLLM to agent frameworks.
The first load fetches WebLLM's runtime and model files; the browser caches them for later loads.

| Package             | Guide                                                     | Examples                                                          | npm                                                    |
| ------------------- | --------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------ |
| `langchain-web-llm` | [Usage and support](packages/langchain-web-llm/README.md) | [LangChain and Deep Agents](examples/langchain-web-llm/README.md) | [npm](https://www.npmjs.com/package/langchain-web-llm) |
| `pi-ai-web-llm`     | [Usage and support](packages/pi-ai-web-llm/README.md)     | [Pi Agent Core and Pi Durable](examples/pi-ai-web-llm/README.md)  | [npm](https://www.npmjs.com/package/pi-ai-web-llm)     |
| `strands-web-llm`   | [Usage and support](packages/strands-web-llm/README.md)   | [Strands SDK and Harness](examples/strands-web-llm/README.md)     | [npm](https://www.npmjs.com/package/strands-web-llm)   |

Model support depends on WebLLM's exact `model_id` and quantization. Each package lists its verified models and features.
Verified examples include Qwen3.5-4B, Qwen3-8B, Gemma 4 E2B, Phi-3.5 Vision, and Snowflake Arctic Embed. Support differs by adapter; see its guide for exact model IDs and features.

For repository development, use [pnpm](https://pnpm.io/installation) and the Node.js version in `.node-version`:

```sh
pnpm install
pnpm check
```
