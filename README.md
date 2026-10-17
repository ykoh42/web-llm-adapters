# WebLLM Adapters

WebLLM adapters for AI agent frameworks. Community maintained.

The first package is [`strands-webllm`](./packages/strands-webllm). It connects a
WebLLM engine to Strands Harness through its SDK model interface, without Vercel
AI SDK or a hosted model API. The [Harness example](examples/harness/README.md)
keeps Harness and its tools in Node.js and runs WebLLM in a controlled browser,
using the package's public `BrowserModel` and `createBrowserBridge` APIs.

## Layout

```text
packages/
  strands-webllm/     Strands model provider and package tests
examples/
  browser/           Browser examples and their GPU/extension tests
  harness/           Node.js Harness example and its integration tests
test/
  browser.ts         Shared Chrome fixture
```

Packages have their own source, build output, and peer dependencies. The root owns
the lockfile, TypeScript settings, linting, formatting, and tests. Shared runtime
code can be added when another adapter needs it. The root pins one SDK runtime
for the Strands package and examples because Harness recognizes models with
`instanceof Model`. The package still declares the SDK as a peer dependency.

## Start

Use Node.js 24 and npm.

```sh
npm ci
npm run build
npm run check
npm run dev
```

Open <http://127.0.0.1:4173>. Click **Load model**, then **Send**. The first load
downloads several GB. Later loads use WebLLM's browser cache. WebGPU and enough
GPU memory are required. FP16 needs `shader-f16`; choose FP32 when it is absent.

The model runs on your device. Model files are downloaded from the URLs in
WebLLM's model catalog. The page starts with Gemma text chat. Choose another
example for JSON, tools, media, or native APIs, and another engine for a Worker.
See the [browser examples](examples/browser/README.md) and [model restrictions](packages/strands-webllm/README.md#limits).
The Gemma tool example runs the tool and returns its result to the model for a final
answer. The Harness example extends the built-in Node.js `read` tool with PDF text
extraction and image OCR, and uses Harness's Exa integration for web search.
Search queries leave your device; model inference remains local.

## Work on a package

```sh
npm run build --workspace strands-webllm
npm run typecheck --workspace strands-webllm
npm test -- packages/strands-webllm/test
```

`npm run dev` builds the package once, then starts Vite. Rebuild the package after
editing its source, or run `npx tsc -p packages/strands-webllm/tsconfig.build.json
--watch` in a second terminal.

`npm run check` runs formatting, linting, all builds, all workspace type checks,
test coverage, unused code checks, and package checks. CI runs this same command
and the isolated installation check.
Run `npm run build` before a standalone `npm run typecheck` so
examples can resolve workspace packages through their real exports.

### Provider flow

Inside `packages/strands-webllm/src`, `model.ts` chooses native chat or the Gemma
protocol. `request.ts` applies shared generation settings; each protocol builds
its own messages or prompt. `chat.ts` and `gemma/stream.ts` turn native output into
the small `StreamChunk` type, then `stream.ts` writes Strands events.

`call.ts` owns the native iterator and its cleanup. `queue.ts` gives each engine
one active call at a time. Keep ownership there: cancellation must close the native
iterator even when Strands stops reading, and the next call must wait for cleanup.
`browser/` carries the same provider across a page boundary. Engine loading, tools,
OCR, and application policy belong to the caller or examples.

## Quality checks

| Command                  | What it checks                                                                          |
| ------------------------ | --------------------------------------------------------------------------------------- |
| `npm test`               | Fast unit and Agent integration tests                                                   |
| `npm run test:coverage`  | Untested code and branches; HTML report in `coverage/index.html`                        |
| `npm run check:unused`   | Unused files, exports, and dependencies with Knip                                       |
| `npm run test:install`   | Pack, install outside the workspace, check public types, and run SDK/Harness tool loops |
| `npm run check:packages` | Package contents with publint, then TypeScript imports with Are the Types Wrong?        |

Coverage must stay at least 90% of lines and 85% of branches across package source.
Coverage is a check for missing tests, not proof that assertions are useful. Tests
also hold back stream chunks, cancel an active call, and check cleanup after errors.

Package checks run after building. Each library adds a `check:package` script;
the root finds those scripts through the workspaces. `strands-webllm` uses the
`esm-only` type-check profile because it ships ESM, not a CommonJS build.

## Real model test

Install Google Chrome, then run:

```sh
npm run test:browser
```

This opens Chrome and runs the browser examples with real model inference.
It can take several minutes and downloads Gemma, Hermes, Phi vision, and Snowflake
model files, including Qwen3, on the first run. To run only the basic Gemma check, use
`npm run test:browser -- examples/browser/test/gemma.spec.ts`.
The test keeps a browser profile in `.cache/chrome` so later runs can reuse model
files. Set `HEADLESS=1` to run without a window if your system supports headless
WebGPU. This test runs separately from CI because it needs a GPU and large model
downloads.

The tests cover the adapter's Strands behavior, actual `createHarness()` calls,
constrained output, sampling, thinking, media, multiple loaded models, custom
processors, integrity checks, and native APIs. CPU tests use a mocked engine;
browser tests run real models. Synthetic silence checks audio decoding and
inference, not transcription quality. Save/resume checks interruption and recovery
after destroying the page during generation.
The Gemma Harness test reads a random code from a temporary file, checks the actual
tool result, and requires Gemma's final answer to contain that code. It also checks
a later conversation turn. No hook supplies the answer or ends the tool turn.
The thinking check also observes actual Worker prompts to verify that Gemma's
reasoning survives the tool call and is stripped before the next user turn.
Use `--think --tool read` in the Harness example, or choose **Gemma thinking + tools**
in the browser app.

For the extension checks, install Playwright's Chromium and run:

```sh
npx playwright install chromium
npm run test:extensions
```

This uses `.cache/chromium-extensions`, loads the example extension and a pinned
[Cross-Origin Storage extension](https://github.com/web-ai-community/cross-origin-storage-extension),
and disables its optional telemetry. The cache check uses two localhost origins
and blocks a second download of the model weights. These checks need a window,
WebGPU, and additional model storage. They run separately from CPU-only CI.

Package tests live under `packages/<name>/test/`; example integration tests live
under `examples/<name>/test/`. The root holds only the shared Chrome fixture and
test runner settings. Each new adapter keeps its tests in its own workspace.
Target versions are WebLLM **0.2.86**, Strands SDK **1.21.0**, and Harness **0.3.0**.
These checks cover specific models and Harness flows; they do not guarantee a
model's tool-selection or answer accuracy for every task.

## Before a release

```sh
npm run check:release
```

This runs the CPU, isolated installation, real GPU, and extension checks in order.
Use the same commit you intend to publish, with Chrome, Playwright Chromium, and
WebGPU available. A failure stops the command. These GPU checks remain local;
GitHub CI runs CPU and installation checks.

`test:install` packs the library into a temporary directory outside this repository,
installs only its peers and Harness, and checks browser and Node.js public types.
It then imports the packed exports and runs actual SDK/Harness tool loops with a
mocked inference endpoint. Real inference is checked by the GPU suite. The temporary
project is removed on success or failure. No workspace symlinks or source imports
are used by the installed consumer.

Choose a license and release version before removing the package's `private` flag.
The release check does not publish to npm.

## Add an adapter

1. Add `packages/<package-name>` with its own `package.json` and `src/index.ts`.
2. Extend `../../tsconfig.base.json` and use a separate build config for `src`.
3. Put framework and WebLLM versions in that package's peer dependencies and dev
   dependencies. Keep framework code inside its adapter.
4. Add tests in `test/*.test.ts` inside the package. The root test runner finds them.
   GPU integration tests use `test/**/*.browser.ts`; example UI tests use
   `examples/<name>/test/*.spec.ts`.
5. Run `npm install` at the root to update the workspace links and lockfile.

Packages and examples are found through `packages/*` and `examples/*` workspaces.
Builds follow workspace order; if packages later depend on each other, put their
dependencies first or add a build tool that orders the graph.

## Code style

Use plain names, short functions, and flat control flow. Keep each function focused
on one job. Prefer WebLLM's existing API to another wrapper. Add shared code only
when a second package needs it. Tests should check behavior through real Strands
interfaces, with a real model check for the browser path.

## Status

This is a public community project. `strands-webllm` is marked `private` and has
not been published to npm. Choose a license and release version, review the model
restrictions, and remove the package's `private` flag before publishing.

References: [Strands issue #2481](https://github.com/strands-agents/harness-sdk/issues/2481),
[WebLLM](https://github.com/mlc-ai/web-llm),
[WebLLM 0.2.86 release](https://github.com/mlc-ai/web-llm/releases/tag/v0.2.86).
