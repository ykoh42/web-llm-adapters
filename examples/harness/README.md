# Strands Harness + WebLLM

Harness runs in Node.js. WebLLM runs in a browser with WebGPU. `BrowserModel`
forwards model requests to the browser's `WebLLMModel` and brings its Strands
stream events back. Harness executes the tools in Node.js. No model API server
or API key is used.

From the repository root:

```sh
npm ci
npm run dev
```

In another terminal:

```sh
npm run start --workspace @webllm-adapters/harness -- 'Say hello.'
npm run start --workspace @webllm-adapters/harness -- --think --tool read 'Read /absolute/path/to/note.txt and summarize it.'
```

This opens Google Chrome, loads Gemma 4 E2B, and reuses `.cache/chrome`. The app
must be running at `http://127.0.0.1:4173`. Close other tests using that profile
before starting the CLI. The first load downloads several GB.

`--think` enables thinking. `--tool read` selects `read` for the first model call
using Strands's standard named tool choice. Gemma writes the arguments, receives
the actual tool result, and writes the final answer. Later calls use auto choice.
E2B does not reliably select tools from the full default tool set, so the named
choice is useful for a specific first step. It supplies no argument values or
completion message and uses no forced end-turn hook. E2B can alter identifiers
in generated answers or miss facts in dense repetitive history, even when a
summary preserves them. Use the actual tool result when an exact value matters.

The example leaves sessions, skills, memory, context management, built-in plugins,
background tasks, subagents, and the default text tools enabled. `printer: false`
keeps SDK printing out of the CLI's final output. `read.media: false` is required
because this bridge carries text, reasoning, and tool JSON. Browser media and
other native APIs have [separate examples](../browser/README.md).

Gemma 4 E2B handles the main conversation, summaries, `web_fetch`, and memory.
Only one set of weights is loaded. The main, auxiliary, and memory models are
request configurations sharing that engine and queue, with a 16,384-token window.
The auxiliary model uses WebLLM's native chat path for tool-free summaries.
[SummaryModel](./summary-model.ts) reads long Harness context summaries in small
parts, then combines the actual model summaries. Gemma missed identifiers when
reading the entire repetitive transcript at once. This costs extra inference
calls, keeps words intact, and leaves ordinary requests unchanged. Empty or
unfinished partial summaries fail instead of being silently combined.
Memory extraction uses the native JSON API. [createMemory](./memory.ts) sets up
that file store while keeping the Harness memory policy.

The auxiliary and memory requests omit the thinking option. Setting it to either
true or false selects Gemma protocol mode, which does not combine with JSON
constraints. The main agent can still use thinking and tools.

After the page has loaded Gemma into its engine, the important setup is
below. [main.ts](./main.ts) has the complete runnable setup.

```ts
const modelId = 'gemma-4-E2B-it-q4f16_1-MLC'
const config = { modelId, contextWindowLimit: 16384 }
const model = new BrowserModel(page, {
  ...config,
  maxTokens: 4096,
  temperature: 1,
  topP: 0.95,
  request: { repetition_penalty: 1.05, extra_body: { enable_thinking: true } },
})
const auxModel = new SummaryModel(page, {
  ...config,
  maxTokens: 512,
  temperature: 0,
})
const options = {
  model,
  auxModel,
  memory: { stores: [createMemory(page, modelId, 16384)] },
  builtinTools: { read: { media: false }, web_fetch: { model: auxModel } },
  printer: false,
}
const agent = await createHarness(options)
try {
  const result = await agent.invoke('Your task')
  console.log(
    result.lastMessage.content
      .filter((b) => b.type === 'textBlock')
      .map((b) => b.text)
      .join('\n'),
  )
} finally {
  await agent.shutdown()
}
```

Thinking stays separate from answer text in Strands events and conversation
history. Thoughts are retained between tools in the current turn and omitted
from the next user turn's model prompt. `String(result)` also includes reasoning;
select text blocks when printing only the final answer.

State is saved under `.agent/`, which Git ignores. Pass
`session: { id: previousSessionId }` to restore a conversation. Memory is separate
from sessions and is searched for a new session too. Put skills under
`.agent/skills/<name>/SKILL.md`, or set `skills` to another directory. Always shut
agents down before closing the browser so pending memory extraction can finish.

Multiple agents can share these configurations and the browser page:

```ts
const agents = await Promise.all([createHarness(options), createHarness(options)])
try {
  const results = await Promise.all(agents.map((agent, i) => agent.invoke(`Task ${i + 1}`)))
} finally {
  await Promise.all(agents.map((agent) => agent.shutdown()))
}
```

The agents also share the memory store from `options`, including its write locks.
Each request has its own stream and cancellation signal. Requests to one engine
run in order; cancelling a queued request leaves the active request alone. Independent engines have
independent queues. This is concurrent request handling, not simultaneous GPU
execution on one engine.

Harness's provider prompt-cache setting does not apply to a pre-built Model.
WebLLM keeps its own model cache and inference state. Native `web_search` is a
provider feature and is not exposed for this custom model. Harness can explicitly
use its optional Exa search integration instead.

Run the real GPU checks with:

```sh
npm run test:browser -- examples/harness/test
```

Tests create isolated temporary files, sessions, skills, and memory. Some use
named choice to test a feature separately from E2B's tool-selection accuracy.
They check actual tool results and final text; no hook supplies an answer.
The older Hermes check remains separate and ends after its tool batch.
Target versions are Harness 0.3.0, Strands SDK 1.21.0, and WebLLM 0.2.86.
