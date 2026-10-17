# Strands Harness + WebLLM

Harness and tools run in Node.js; WebLLM inference runs in a WebGPU browser.
This example uses the package's public `BrowserModel` and `createBrowserBridge`.

## Run

```sh
npm ci
npm run dev
```

In another terminal:

```sh
npm run start --workspace @webllm-adapters/harness -- 'Say hello.'
npm run start --workspace @webllm-adapters/harness -- --think --tool read 'Read /absolute/path/to/note.txt and summarize it.'
npm run start --workspace @webllm-adapters/harness -- --tool read 'Read /absolute/path/to/note.pdf and summarize its text.'
npm run start --workspace @webllm-adapters/harness -- --think --tool read 'Read the text in /absolute/path/to/note.png.'
npm run start --workspace @webllm-adapters/harness -- --tool web_search 'Find the official Strands Harness SDK repository.'
```

The CLI opens Chrome, loads Gemma 4 E2B, and reuses `.cache/chrome`.
Keep the app running at <http://127.0.0.1:4173>. The first run downloads model files.

| Option          | Meaning                                                                   |
| --------------- | ------------------------------------------------------------------------- |
| `--think`       | Keep thinking separate from answer text                                   |
| `--tool <name>` | Select the first tool; the model generates its arguments and final answer |

E2B may select the wrong tool or alter identifiers. Named choice helps select a
specific first step; use the actual tool result when exact values matter.

## Examples

| Feature                  | Source                                          | Notes                                                     |
| ------------------------ | ----------------------------------------------- | --------------------------------------------------------- |
| Full CLI setup           | [main.ts](main.ts)                              | Browser startup, model config, Harness, shutdown          |
| Text, PDF, image reading | [read.ts](read.ts)                              | Extends Harness `read`; text paging retained              |
| Image OCR                | [ocr.ts](ocr.ts)                                | English/Korean PNG and JPEG; local CPU processing         |
| Summaries                | [summary-model.ts](summary-model.ts)            | Reads long context in parts using the same weights        |
| Memory                   | [memory.ts](memory.ts)                          | Harness memory policy with native JSON extraction         |
| Native APIs in tools     | [engine.browser.ts](test/engine.browser.ts)     | Embeddings, raw completions, and generation recovery      |
| Restart recovery         | [recovery.browser.ts](test/recovery.browser.ts) | Restore WebLLM checkpoints and Harness history separately |

Sessions, skills, memory, context management, background tasks, subagents, and
ordinary tools keep their Harness defaults. `read: false` replaces the built-in
reader with the extended tool under the same name. `printer: false` keeps the CLI
output focused on the answer. Main, auxiliary, and memory requests share one engine.

State lives under `.agent/`. Pass `session: { id: previousSessionId }` to restore a
conversation. Put skills under `.agent/skills/<name>/SKILL.md`. Shut agents down
before closing the browser so pending memory work can finish.

## Limits

| Feature              | Limit                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------- |
| PDF                  | Embedded text only; 20 MiB; scanned pages need OCR                                      |
| OCR                  | PNG/JPEG up to 20 MiB; accuracy depends on image quality                                |
| Gemma images         | The WebLLM artifact has no image input; this example uses OCR                           |
| Speech transcription | Not enabled here; real speech checks were inaccurate                                    |
| Gemma thinking       | Use thinking off for large tool previews; E2B can repeat thinking until the token limit |
| Web search           | Exa receives search queries; optional `EXA_API_KEY` increases limits                    |
| Concurrent requests  | One shared engine queue; independent histories and cancellation                         |
| Recovery             | Native generation checkpoints and Harness sessions store different state                |

The [package support table](../../packages/strands-webllm/README.md#support) covers
provider features. The [browser examples](../browser/README.md) cover media and
native engine APIs. No hook supplies an answer or completes the model's tool turn.

## Test

```sh
npm run test:browser -- examples/harness/test
```

These tests run real inference with isolated files, sessions, skills, and memory.
Some use named choice to test a feature independently of tool-selection accuracy.
