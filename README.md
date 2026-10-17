# WebLLM Adapters

Community packages that connect WebLLM to agent frameworks through their public APIs.

| Package           | Guide                                                   | Examples                                                      |
| ----------------- | ------------------------------------------------------- | ------------------------------------------------------------- |
| `strands-web-llm` | [Usage and support](packages/strands-web-llm/README.md) | [Strands SDK and Harness](examples/strands-web-llm/README.md) |

## Development

Use Node.js 24. Each package owns its source and unit tests. Its examples and
integration tests live under `examples/<package>/`. The root shares tooling.

```text
packages/strands-web-llm/     Library and unit tests
examples/strands-web-llm/    Browser, Harness, and integration tests
```

```sh
npm ci
npm run check
npm run test:install
```

Real inference checks require Chrome and a WebGPU GPU:

```sh
npm run test:browser
npm run test:extensions
```

For extension checks, install Chromium with `npx playwright install chromium`.
`npm run check:release` runs every check above. CI runs CPU and installation checks.
Coding rules are in [AGENTS.md](AGENTS.md).

## Release

MIT · 0.1.0. Not published yet.
