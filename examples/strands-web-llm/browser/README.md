# Strands browser examples

A local demo and real-model test page for `strands-web-llm`. It is not required to use the package.

From the repository root:

```sh
npm ci
npm run dev:strands
```

Open <http://127.0.0.1:4173>. Choose an **Example** and **Engine**, click **Load model**,
then **Send**. Web Worker is selected by default to keep inference off the main thread.
**Unload** before changing either. Each run starts a new conversation.
The first load fetches WebLLM's runtime and model files; later runs use the browser cache.

| Example                     | Source                                                                   | Purpose                                                        |
| --------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------- |
| Chat                        | [chat.ts](src/examples/chat.ts)                                          | Streaming, usage, and cancellation                             |
| Tools and thinking          | [tools.ts](src/examples/tools.ts)                                        | Call a tool and answer from its result                         |
| Thinking modes and sampling | [advanced.ts](src/examples/advanced.ts)                                  | Separate reasoning; compare sampling and timing                |
| JSON and grammar            | [chat.ts](src/examples/chat.ts), [advanced.ts](src/examples/advanced.ts) | JSON schema, token probabilities, grammar, and structural tags |
| Images and audio            | [media.ts](src/examples/media.ts)                                        | Image bytes, WAV, and PCM input                                |

**Response** shows answer text; **Details** shows reasoning, tools, and metadata.
Choose models from the package's [verified model table](../../../packages/strands-web-llm/README.md#models).
The demo already configures Gemma's [required runtime](../../../shared/runtime/README.md).
Audio examples check input handling; they do not establish speech transcription accuracy.

<details>
<summary>Chrome extension</summary>

Selecting Cross-origin Storage switches the engine to Main thread mode.
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
