# WebLLM Adapters

Community adapters that connect WebLLM to agent frameworks.
The first package, [strands-webllm](packages/strands-webllm/README.md), connects
WebLLM to Strands SDK and Harness through their public APIs.

## Start

Use Node.js 24, Google Chrome, and a GPU with WebGPU support.

```sh
npm ci
npm run dev
```

Open <http://127.0.0.1:4173>, choose an example, load its model, and send a message.
The first load downloads model files; later runs reuse the browser cache.

| Read                                               | Purpose                                       |
| -------------------------------------------------- | --------------------------------------------- |
| [Package guide](packages/strands-webllm/README.md) | API, quick start, and support table           |
| [Browser examples](examples/browser/README.md)     | Chat, tools, thinking, media, and native APIs |
| [Harness example](examples/harness/README.md)      | Node.js tools with browser inference          |

## Development

Each package keeps its own source, build, dependencies, and tests. Examples have
separate integration tests. The root owns shared tools and the lockfile.

```text
packages/strands-webllm/  Provider and package tests
examples/browser/        Browser examples and GPU tests
examples/harness/        Harness example and integration tests
test/                   Shared browser fixture
```

| Command                   | Checks                                                                              |
| ------------------------- | ----------------------------------------------------------------------------------- |
| `npm run check`           | Format, lint, build, types, CPU tests, coverage, unused code, package contents      |
| `npm run test:install`    | Packed package installed outside this workspace; public types and SDK/Harness loops |
| `npm run test:browser`    | Real GPU inference in Chrome                                                        |
| `npm run test:extensions` | Extension worker and cross-origin cache; needs Playwright Chromium                  |
| `npm run check:release`   | All checks above, in order; does not publish                                        |

CI runs `check` and `test:install`. CPU tests mock inference; GPU tests use real
models. Install extension-test Chromium with `npx playwright install chromium`.
Close other processes using `.cache/chrome` before running browser tests.

After editing library source, run `npm run build --workspace strands-webllm`.
To add another adapter, create `packages/<name>` with its own config and tests;
share code only when another package needs it. Coding rules are in [AGENTS.md](AGENTS.md).

## Status

The repository is public; the package is not published to npm. A license and
first release version still need to be chosen. Run `npm run check:release` on the
commit to publish before removing the package's `private` flag.

Targets: WebLLM **0.2.86**, Strands SDK **1.21.0**, Harness **0.3.0**.
