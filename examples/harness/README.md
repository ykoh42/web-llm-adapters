# Strands Harness + WebLLM

Harness runs in Node.js. WebLLM runs in a browser with WebGPU. This example uses
Playwright to connect them: `BrowserModel` forwards model requests to the browser's
`WebLLMModel` and streams its Strands events back. Harness owns conversation history
and executes its tools in Node.js. No model API server or API key is used.

From the repository root, start the browser app:

```sh
npm ci
npm run dev
```

In another terminal, run the Harness example:

```sh
npm run start --workspace @webllm-adapters/harness -- 'Say hello.'
```

This opens Google Chrome and loads Gemma 4 E2B. The first load downloads the model.
It reuses `.cache/chrome`; close other tests using that profile before starting it.
The local app must be running at `http://127.0.0.1:4173`.

To use Harness's built-in Node.js `read` tool, choose the Hermes mode and pass a
prompt containing the absolute path of a file you want the agent to read:

```sh
npm run start --workspace @webllm-adapters/harness -- --read 'Use read to read /absolute/path/to/note.txt.'
```

WebLLM's Hermes tool mode requires its own system prompt and a tool JSON grammar.
The example disables the Harness prompt and ends after the tool batch, displaying
the actual tool result. Gemma does not accept native tools in WebLLM 0.2.86.

The example explicitly configures the tools it uses and turns off background task
tools, memory, sessions, skills, context management, and built-in plugins. It does
not demonstrate every Harness default. The browser bridge is a small local example
for text and tool JSON, not a network service or a separate published provider.
Browser-only native APIs and media are demonstrated in [the browser app](../browser/README.md).

`npm run test:browser -- examples/harness/test` runs real Harness checks for
streaming, history, cancellation/reuse, and a built-in file read. These use Harness
0.3.0, Strands SDK 1.21.0, and WebLLM 0.2.86. The file test creates its own temporary
note and checks its content in the real tool result.
