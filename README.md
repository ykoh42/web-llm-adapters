# WebLLM Adapters

WebLLM adapters for AI agent frameworks. Community maintained.

The first package is [`strands-webllm`](./packages/strands-webllm). It connects a
WebLLM engine to Strands Harness through its SDK model interface, without Vercel
AI SDK or a hosted model API. The [Harness example](examples/harness/README.md)
keeps Harness and its tools in Node.js and runs WebLLM in a controlled browser.

## Layout

```text
packages/
  strands-webllm/     Strands model provider
examples/
  browser/           Strands and native WebLLM examples
  harness/           Node.js Harness with browser inference
test/
  browser/           Real GPU model checks
  extensions/        Real extension worker and shared-cache checks
```

Packages have their own source, build output, and peer dependencies. The root owns
the lockfile, TypeScript settings, linting, formatting, and tests. Shared runtime
code can be added when another adapter needs it.

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
Gemma does not accept native tool calls in this WebLLM release; that example uses Hermes.

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
test coverage, unused code checks, and package checks. CI runs this same command.
Run `npm run build` before a standalone `npm run typecheck` so
examples can resolve workspace packages through their real exports.

## Quality checks

| Command                  | What it checks                                                                   |
| ------------------------ | -------------------------------------------------------------------------------- |
| `npm test`               | Fast unit and Agent integration tests                                            |
| `npm run test:coverage`  | Untested code and branches; HTML report in `coverage/index.html`                 |
| `npm run check:unused`   | Unused files, exports, and dependencies with Knip                                |
| `npm run check:packages` | Package contents with publint, then TypeScript imports with Are the Types Wrong? |

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
`npm run test:browser -- test/browser/gemma.spec.ts`.
The test keeps a browser profile in `.cache/chrome` so later runs can reuse model
files. Set `HEADLESS=1` to run without a window if your system supports headless
WebGPU. This test runs separately from CI because it needs a GPU and large model
downloads.

The tests cover the adapter's Strands behavior, actual `createHarness()` calls,
constrained output, sampling, thinking, media, multiple loaded models, custom
processors, integrity checks, and native APIs. CPU tests use a mocked engine;
browser tests run real models. Synthetic silence checks audio decoding and
inference, not transcription quality. Save/resume checks an orderly interruption.

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

Verification is recorded in executable tests under `test/` and `examples/harness/test/`.
Target versions are WebLLM **0.2.86**, Strands SDK **1.21.0**, and Harness **0.3.0**.
Compatibility with these versions does not establish support for every model or
for every Harness default.

## Add an adapter

1. Add `packages/<package-name>` with its own `package.json` and `src/index.ts`.
2. Extend `../../tsconfig.base.json` and use a separate build config for `src`.
3. Put framework and WebLLM versions in that package's peer dependencies and dev
   dependencies. Keep framework code inside its adapter.
4. Add tests in `test/*.test.ts` inside the package. The root test runner finds them.
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
