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

To use Harness's built-in Node.js `read` tool, pass `--read` and a
prompt containing the absolute path of a file you want the agent to read:

```sh
npm run start --workspace @webllm-adapters/harness -- --read 'Read /absolute/path/to/note.txt using read, then summarize it.'
```

Both commands use Gemma 4 E2B. In `--read` mode, Gemma selects the tool, Harness
reads the file in Node.js, and Gemma receives the result and writes the final
answer. The example keeps Harness's system prompt and has no forced end-turn hook.
The adapter uses Gemma's native function-call format through raw completions,
because WebLLM 0.2.86 restricts its separate `tools` API to Hermes.

Add `--think` to enable Gemma thinking, with or without `--read`:

```sh
npm run start --workspace @webllm-adapters/harness -- --think --read 'Read /absolute/path/to/note.txt using read, then summarize it.'
```

Thinking streams as Strands reasoning events and stays in Agent history. The
model prompt retains it between tool calls and strips old thoughts at the next
user turn. The CLI prints only the model's final text blocks. It adds no completion
message or forced end-turn hook. The browser example shows reasoning separately
in **Details**.

The example explicitly configures the tools it uses and turns off background task
tools, memory, sessions, skills, context management, and built-in plugins. It does
not demonstrate every Harness default. The browser bridge is a small local example
for text, reasoning, and tool JSON, not a network service or a separate published provider.
Browser-only native APIs and media are demonstrated in [the browser app](../browser/README.md).

`npm run test:browser -- examples/harness/test` runs real Harness checks for
streaming, history, cancellation/reuse, and a built-in file read. These use Harness
0.3.0, Strands SDK 1.21.0, and WebLLM 0.2.86. The file test creates its own temporary
note with a random code and literal Gemma control markers. The Gemma check requires
a successful read, exactly two model calls, a final model answer containing the
code, and recall in the next conversation turn. A separate legacy Hermes check
still ends after its tool batch.
A second Gemma check sends nested arguments, passes a random code from one tool to
another, and requires the final answer to use the second tool's random receipt.
The thinking check observes the actual Worker completion requests: it requires
thinking before the real file read, a final answer using the file's random code,
preserved thoughts after the tool result, and stripped thoughts at the next user
turn. It also cancels a reasoning delta and checks that the engine can run again.
