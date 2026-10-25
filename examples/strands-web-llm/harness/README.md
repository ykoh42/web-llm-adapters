# Strands Harness + WebLLM

Harness and tools run in Node.js; WebLLM runs in a WebGPU browser.
Tools use your account's file and shell permissions; configure a Harness sandbox to isolate them.

## Run

From the repository root, start the local browser host:

```sh
pnpm install
pnpm dev:strands
```

Keep it running at <http://127.0.0.1:4173>. In another terminal:

```sh
pnpm --filter @web-llm-adapters/strands-harness start 'Say hello.'
```

The CLI opens Chrome and caches model downloads. Gemma 4 E2B is the example default;
use `--model <exact model_id>` to choose another [verified model](../../../README.md#models).
The demo includes Gemma's required runtime setup.

| Option              | Purpose                                                                   |
| ------------------- | ------------------------------------------------------------------------- |
| `--model <id>`      | Choose the loaded WebLLM model                                            |
| `--think`           | Enable separate reasoning blocks                                          |
| `--semantic-memory` | Load Snowflake Arctic Embed S for semantic memory; uses extra GPU memory  |
| `--tool <name>`     | Select the first tool; the model generates its arguments and final answer |

For example, read a file with thinking enabled:

```sh
pnpm --filter @web-llm-adapters/strands-harness start --think --tool read 'Read /absolute/path/to/note.txt and summarize it.'
```

## Source

| Feature                                          | Source                       |
| ------------------------------------------------ | ---------------------------- |
| Browser startup, model config, Harness, shutdown | [main.ts](src/main.ts)       |
| Text and PDF reading                             | [read.ts](src/tools/read.ts) |
| Local image OCR                                  | [ocr.ts](src/tools/ocr.ts)   |
| Memory and embedding search                      | [memory.ts](src/memory.ts)   |

The example keeps Harness's sessions, skills, background tasks, and subagents enabled.
It replaces `read` with a PDF/OCR reader and supplies WebLLM-backed memory.
State is stored under `.agent/`. Shut agents down before closing the browser.

## Limits

| Feature             | Limit                                                                     |
| ------------------- | ------------------------------------------------------------------------- |
| PDF                 | Embedded text only; 20 MiB; scanned pages need OCR                        |
| Image reading       | Local OCR for PNG/JPEG up to 20 MiB; Gemma has no native image input here |
| Speech              | Transcription is not enabled                                              |
| Web search          | Uses Exa; queries leave the machine; `EXA_API_KEY` is optional            |
| Embedding memory    | 512 tokens per text; scans all entries per search; split long entries     |
| Concurrent requests | One engine queue, with separate histories and cancellation                |
| Recovery            | Harness sessions and native generation checkpoints save different state   |

See the [package support table](../../../packages/strands-web-llm/README.md#support)
for adapter features and the [browser examples](../browser/README.md) for native WebLLM APIs.

<details>
<summary>Run real-model tests</summary>

```sh
pnpm test:browser --project=strands-harness
```

Tests use isolated files, sessions, skills, and memory. Some use named tool choice
to check a feature independently of the model's choice of tool. Native API and
restart checks live under [test/browser](test/browser).

</details>
