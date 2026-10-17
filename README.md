# WebLLM Adapters

[WebLLM](https://github.com/mlc-ai/web-llm) runs AI models locally in your browser using WebGPU.
These community adapters connect WebLLM to agent frameworks.

| Package           | Guide                                                   | Examples                                                      |
| ----------------- | ------------------------------------------------------- | ------------------------------------------------------------- |
| `strands-web-llm` | [Usage and support](packages/strands-web-llm/README.md) | [Strands SDK and Harness](examples/strands-web-llm/README.md) |

## Models

Use WebLLM chat models through Strands. WebLLM 0.2.86's catalog includes:

| Use                       | Models                                                                                                                                 |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Text                      | Llama, Qwen, Gemma, Phi, Hermes, Mistral, Ministral, DeepSeek-R1 distills, SmolLM2, OLMo 2, StableLM, RedPajama, TinyLlama, WizardMath |
| Images                    | Phi-3.5 Vision                                                                                                                         |
| Audio                     | Gemma 4 E2B                                                                                                                            |
| Embeddings via native API | Snowflake Arctic Embed                                                                                                                 |

Features and GPU requirements vary by model; the catalog is not a per-model test guarantee.
See [model IDs and limits](packages/strands-web-llm/README.md#models) or the
[full WebLLM catalog](https://github.com/mlc-ai/web-llm/blob/v0.2.86/src/config.ts).

## Release

MIT · 0.1.0. Not published yet.
