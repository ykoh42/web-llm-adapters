# Strands Harness + WebLLM

Harness and tools run in Node.js; WebLLM inference runs in a WebGPU browser.
This example uses the package's public `BrowserModel` and `createBrowserBridge`.

## Run

```sh
npm ci
npm run dev:strands
```

In another terminal:

```sh
npm run start --workspace @web-llm-adapters/strands-harness -- 'Say hello.'
npm run start --workspace @web-llm-adapters/strands-harness -- --think --tool read 'Read /absolute/path/to/note.txt and summarize it.'
npm run start --workspace @web-llm-adapters/strands-harness -- --tool read 'Read /absolute/path/to/note.pdf and summarize its text.'
npm run start --workspace @web-llm-adapters/strands-harness -- --think --tool read 'Read the text in /absolute/path/to/note.png.'
npm run start --workspace @web-llm-adapters/strands-harness -- --tool web_search 'Find the official Strands Harness SDK repository.'
```

This example selects Gemma 4 E2B; the library accepts other loaded WebLLM models.
The CLI opens Chrome and reuses `.cache/chrome`.
Keep the app running at <http://127.0.0.1:4173>. The first run downloads model files.
The page uses the package's [cache-fixed runtime](../../../packages/strands-web-llm/runtime/README.md)
with WebLLM 0.2.86; no upstream update or runtime build is needed.

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

Sessions, skills, memory, background tasks, subagents, and ordinary tools keep
their Harness defaults.
`read: false` replaces the built-in reader with the extended tool under the same name. `printer: false` keeps the CLI
output focused on the answer. Main, auxiliary, and memory requests share one engine.

State lives under `.agent/`. Pass `session: { id: previousSessionId }` to restore a
conversation. Put skills under `.agent/skills/<name>/SKILL.md`. Shut agents down
before closing the browser so pending memory work can finish.

## Limits

| Feature              | Limit                                                                    |
| -------------------- | ------------------------------------------------------------------------ |
| PDF                  | Embedded text only; 20 MiB; scanned pages need OCR                       |
| OCR                  | PNG/JPEG up to 20 MiB; accuracy depends on image quality                 |
| Gemma images         | The WebLLM artifact has no image input; this example uses OCR            |
| Speech transcription | Not enabled here; real speech checks were inaccurate                     |
| Web search           | Exa receives search queries; optional `EXA_API_KEY` increases limits     |
| Concurrent requests  | One shared engine queue; independent histories and cancellation          |
| Recovery             | Native generation checkpoints and Harness sessions store different state |

The [package support table](../../../packages/strands-web-llm/README.md#support) covers
provider features. The [browser examples](../browser/README.md) cover media and
native engine APIs. No hook supplies an answer or completes the model's tool turn.

## Test

```sh
npm run test:browser -- examples/strands-web-llm/harness/test
```

These tests run real inference with isolated files, sessions, skills, and memory.
Some use named choice to test a feature independently of tool-selection accuracy.
