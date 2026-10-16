# WebLLM Adapters

WebLLM adapters for AI agent frameworks. Community maintained.

The first package is [`strands-webllm`](./packages/strands-webllm). It connects a
WebLLM engine to Strands Agents without Vercel AI SDK or a hosted model API.

## Layout

```text
packages/
  strands-webllm/     Strands model provider
examples/
  browser/           Gemma 4 E2B playground
test/
  browser/           Real model smoke test
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
WebLLM's model catalog. The example uses no tools: WebLLM 0.2.86 does not yet
accept native tool calls for Gemma 4 E2B.

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
and unit tests. Run `npm run build` before a standalone `npm run typecheck` so
examples can resolve workspace packages through their real exports.

## Real model test

Install Google Chrome, then run:

```sh
npm run test:browser
```

This opens Chrome, loads Gemma 4 E2B FP16, sends a message through a real Strands
Agent, checks the response, and unloads the model. It can take several minutes.
The test keeps a browser profile in `.cache/chrome` so later runs can reuse model
files. Set `HEADLESS=1` to run without a window if your system supports headless
WebGPU. This test runs separately from CI because it needs a GPU and large model
downloads.

Verified on 2026-10-16 with WebLLM 0.2.86 and Strands SDK 1.21.0: Gemma 4 E2B FP16
loaded in Chrome, returned `ready` through Strands streaming, and unloaded. Unit
tests also cover an Agent tool loop, cancellation, failed streams, and media mapping.

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

This repository is private. `strands-webllm` is marked `private` too and has not
been published to npm. Choose a license and release version, review the support
matrix, and remove the package's `private` flag before publishing.

References: [Strands issue #2481](https://github.com/strands-agents/harness-sdk/issues/2481),
[WebLLM](https://github.com/mlc-ai/web-llm),
[WebLLM 0.2.86 release](https://github.com/mlc-ai/web-llm/releases/tag/v0.2.86).
