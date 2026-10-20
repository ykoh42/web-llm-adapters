# WebLLM Adapters

[WebLLM](https://github.com/mlc-ai/web-llm) runs AI models locally in your browser using WebGPU.
These community adapters connect WebLLM to agent frameworks.
The first load fetches WebLLM's runtime and model files; the browser caches them for later loads.

| Package             | Guide                                                     | Examples                                                          | npm                                                              |
| ------------------- | --------------------------------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------- |
| `langchain-web-llm` | [Usage and support](packages/langchain-web-llm/README.md) | [LangChain and Deep Agents](examples/langchain-web-llm/README.md) | [0.1.0](https://www.npmjs.com/package/langchain-web-llm/v/0.1.0) |
| `strands-web-llm`   | [Usage and support](packages/strands-web-llm/README.md)   | [Strands SDK and Harness](examples/strands-web-llm/README.md)     | [0.1.1](https://www.npmjs.com/package/strands-web-llm/v/0.1.1)   |

Model support depends on WebLLM's exact `model_id` and quantization. Each package lists its verified models and features.
