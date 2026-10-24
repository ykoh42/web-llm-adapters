# WebLLM Adapters

[WebLLM](https://github.com/mlc-ai/web-llm) runs AI models locally in your browser using WebGPU.
These community adapters connect WebLLM to agent frameworks.
The first load fetches WebLLM's runtime and model files; the browser caches them for later loads.

| Package             | Guide                                                     | Examples                                                          | npm                                                    |
| ------------------- | --------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------ |
| `langchain-web-llm` | [Usage and support](packages/langchain-web-llm/README.md) | [LangChain and Deep Agents](examples/langchain-web-llm/README.md) | [npm](https://www.npmjs.com/package/langchain-web-llm) |
| `pi-ai-web-llm`     | [Usage and support](packages/pi-ai-web-llm/README.md)     | [Pi Agent Core and Pi Durable](examples/pi-ai-web-llm/README.md)  | [npm](https://www.npmjs.com/package/pi-ai-web-llm)     |
| `strands-web-llm`   | [Usage and support](packages/strands-web-llm/README.md)   | [Strands SDK and Harness](examples/strands-web-llm/README.md)     | [npm](https://www.npmjs.com/package/strands-web-llm)   |

## Models

Exact WebLLM **0.2.86** IDs checked with the adapters below. **All** = LangChain, Pi, and Strands.
`O` = verified; `△` = unreliable; `X` = unverified. Tools covers call → result → final answer; Think means separate reasoning blocks.
Audio is verified in LangChain and Strands only. Decode speeds are local M4 Pro (24 GB), Chrome 155 measurements.

| Exact WebLLM `model_id`             |                                                                                 MMLU-Pro | Decode tok/s | Tools | Think | Image | Audio | Video | Verified adapters |
| ----------------------------------- | ---------------------------------------------------------------------------------------: | -----------: | ----- | ----- | ----- | ----- | ----- | ----------------- |
| `Qwen3.5-9B-q4f16_1-MLC`            |                                           [82.5](https://huggingface.co/Qwen/Qwen3.5-9B) |         19.7 | O     | O     | X     | X     | X     | Strands           |
| `Qwen3.5-4B-q4f16_1-MLC`            |                                           [79.1](https://huggingface.co/Qwen/Qwen3.5-4B) |         33.0 | O     | O     | X     | X     | X     | All               |
| `Qwen3-8B-q4f16_1-MLC`              |                 [74.2](https://huggingface.co/nvidia/Qwen3-8B-DMS-8x#evaluation-dataset) |         32.0 | O     | O     | X     | X     | X     | All               |
| `Qwen3-4B-q4f16_1-MLC`              |                   [70.4](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507#performance) |         51.5 | △     | O     | X     | X     | X     | Strands           |
| `Qwen3.5-2B-q4f16_1-MLC`            |                                           [66.5](https://huggingface.co/Qwen/Qwen3.5-2B) |         59.7 | O     | O     | X     | X     | X     | Strands           |
| `gemma-4-E2B-it-q4f16_1-MLC`        |                                     [60.0](https://huggingface.co/google/gemma-4-E2B-it) |         38.5 | O     | O     | X     | O     | X     | All               |
| `Qwen2.5-7B-Instruct-q4f16_1-MLC`   |                                       [56.3](https://qwenlm.github.io/blog/qwen2.5-llm/) |         38.0 | O     | X     | X     | X     | X     | Strands           |
| `Qwen2.5-3B-Instruct-q4f16_1-MLC`   |                                       [43.7](https://qwenlm.github.io/blog/qwen2.5-llm/) |         59.5 | O     | X     | X     | X     | X     | All               |
| `Qwen3.5-0.8B-q4f16_1-MLC`          |                                         [42.3](https://huggingface.co/Qwen/Qwen3.5-0.8B) |         71.2 | O     | O     | X     | X     | X     | Strands           |
| `Qwen3-0.6B-q4f16_1-MLC`            | [38.27](https://huggingface.co/MultiverseComputingCAI/LittleLamb#evaluation--benchmarks) |         90.4 | △     | O     | X     | X     | X     | Strands           |
| `Qwen2.5-1.5B-Instruct-q4f16_1-MLC` |                                       [32.4](https://qwenlm.github.io/blog/qwen2.5-llm/) |         87.6 | O     | X     | X     | X     | X     | Strands           |
| `Qwen2.5-0.5B-Instruct-q4f16_1-MLC` |                                       [15.0](https://qwenlm.github.io/blog/qwen2.5-llm/) |        136.8 | O     | X     | X     | X     | X     | Strands           |

Qwen3 0.6B can miscopy arguments or omit answers with thinking on; Qwen3 4B can call dependent tools together.
Gemma needs the bundled [runtime setup](shared/runtime/README.md). Audio works in ordinary chat, without tools or thinking. Video is unsupported.

<details>
<summary>Benchmark notes and tool formats</summary>

Models are ranked by published upstream MMLU-Pro with thinking on; Qwen2.5 has no thinking mode.
Sources and evaluation settings vary. Decode speeds are the median of three local runs,
thinking off, max 128 tokens; they are not portable across devices.

Tool formats follow [Hermes](https://github.com/QwenLM/Qwen3/blob/main/docs/source/framework/function_call.md)
(Qwen2.5/Qwen3), [Qwen3 Coder](https://github.com/vllm-project/vllm/blob/5c1c903501927ea7c6c699aabd4c271015e66e71/vllm/parser/qwen3.py)
(Qwen3.5), and [Gemma](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4).

</details>

### Images and embeddings

| Exact WebLLM `model_id`                  | Use                        | Verified adapters  |
| ---------------------------------------- | -------------------------- | ------------------ |
| `Phi-3.5-vision-instruct-q4f16_1-MLC`    | Image input                | All                |
| `snowflake-arctic-embed-m-q0f32-MLC-b32` | Embeddings, 768 dimensions | LangChain, Strands |
| `snowflake-arctic-embed-m-q0f32-MLC-b4`  | Embeddings, 768 dimensions | LangChain, Strands |
| `snowflake-arctic-embed-s-q0f32-MLC-b32` | Embeddings, 384 dimensions | LangChain, Strands |
| `snowflake-arctic-embed-s-q0f32-MLC-b4`  | Embeddings, 384 dimensions | LangChain, Strands |

Embedding inputs are limited to 512 tokens. No listed model is verified for both tools and vision.

## Development

For repository development, use [pnpm](https://pnpm.io/installation) and the Node.js version in `.node-version`:

```sh
pnpm install
pnpm check
```

CI also tests packed consumer installs. Run `pnpm test:install:manager <manager>` with `yarn-pnp`, `yarn-node-modules`, or `pnpm`.
