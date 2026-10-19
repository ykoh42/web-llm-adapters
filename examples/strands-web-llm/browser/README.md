# Strands browser examples

A local demo and real-model test page for `strands-web-llm`. It is not required to use the package.

From the repository root:

```sh
npm ci
npm run dev:strands
```

Open <http://127.0.0.1:4173>. Choose an **Example** and **Engine**, click **Load model**,
then **Send**. **Unload** before changing either. Each run starts a new conversation.
The first load downloads model files; later runs use the browser cache.

| Example                     | Source                                                                   | Purpose                                                        |
| --------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------- |
| Chat                        | [chat.ts](src/examples/chat.ts)                                          | Streaming, usage, and cancellation                             |
| Tools and thinking          | [tools.ts](src/examples/tools.ts)                                        | Call a tool and answer from its result                         |
| Thinking modes and sampling | [advanced.ts](src/examples/advanced.ts)                                  | Separate reasoning; compare sampling and timing                |
| JSON and grammar            | [chat.ts](src/examples/chat.ts), [advanced.ts](src/examples/advanced.ts) | JSON schema, token probabilities, grammar, and structural tags |
| Images and audio            | [media.ts](src/examples/media.ts)                                        | Image bytes, WAV, and PCM input                                |
| Native WebLLM APIs          | [native.ts](src/examples/native.ts)                                      | Embeddings, raw completions, and generation save/resume        |

**Response** shows answer text; **Details** shows reasoning, tools, and metadata.
Choose models from the package's [verified model table](../../../packages/strands-web-llm/README.md#models).
The demo already configures Gemma's [required runtime](../../../shared/runtime/README.md).
Audio examples check input handling; they do not establish speech transcription accuracy.

<details>
<summary>Engine and extension checks</summary>

Open <http://127.0.0.1:4173/engine.html> for WebLLM setup and compatibility checks:

| Check                                     | Source                                  |
| ----------------------------------------- | --------------------------------------- |
| Multiple loaded models                    | [models.ts](src/checks/models.ts)       |
| Custom token sampling                     | [processor.ts](src/checks/processor.ts) |
| Progress, GPU info, history, stats, reset | [state.ts](src/checks/state.ts)         |
| Config, Wasm, and tokenizer integrity     | [integrity.ts](src/checks/integrity.ts) |

The Hermes tool batch checks WebLLM's native tool API; it ends after tool execution.
The integrity check computes hashes; applications should pin trusted hashes.
Cache deletion requires unloading. Cross-origin cache uses the Cross-Origin Storage extension
in Main thread mode; shared deletion is unavailable.

To try the Chrome extension worker, run `npm run build`, open `chrome://extensions`,
enable Developer mode, and load `examples/strands-web-llm/browser/dist` as an unpacked extension.
Open its `index.html` page and choose **Chrome extension worker**.

</details>

<details>
<summary>Run real-model tests</summary>

```sh
npm run test:browser -- --project=strands-browser
```

These checks use Chrome and a real WebGPU device. For extension checks:

```sh
npx playwright install chromium
npm run test:extensions
```

Model files and GPU memory must fit on the machine.

</details>
