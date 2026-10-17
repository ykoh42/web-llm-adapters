# Strands Harness + WebLLM

Harness runs in Node.js. WebLLM runs in a browser with WebGPU. The package exports
`BrowserModel` and `createBrowserBridge`; this example supplies browser startup and
application settings. `BrowserModel` forwards model requests to the browser's
`WebLLMModel` and brings its Strands stream events back. Harness executes the tools in Node.js. No model API server
or model API key is used. Web search uses Harness's Exa integration.

From the repository root:

```sh
npm ci
npm run dev
```

In another terminal:

```sh
npm run start --workspace @webllm-adapters/harness -- 'Say hello.'
npm run start --workspace @webllm-adapters/harness -- --think --tool read 'Read /absolute/path/to/note.txt and summarize it.'
npm run start --workspace @webllm-adapters/harness -- --tool read 'Read /absolute/path/to/note.pdf and summarize its text.'
npm run start --workspace @webllm-adapters/harness -- --think --tool read 'Read the text in /absolute/path/to/note.png and summarize it.'
npm run start --workspace @webllm-adapters/harness -- --tool web_search 'Find the official Strands Harness SDK repository.'
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
summary preserves them. It can also reuse an older value after a file changes,
even when the new tool result contains the current value. Use the actual tool
result when an exact value matters.

The example leaves sessions, skills, memory, context management, built-in plugins,
background tasks, subagents, and the default text tools enabled. `printer: false`
keeps SDK printing out of the CLI's final output. [createRead](./read.ts) extends
the built-in text reader with local PDF text extraction and image OCR.
`read: false` replaces the built-in instance with this extended tool under the
same name; text paging still uses Harness's reader.

PDF reading extracts embedded text without a model. PDF files are limited to
20 MiB. Input files are read through the agent's sandbox. Scanned PDF pages and
Office files need another extraction tool.
WAV transcription is not enabled here: speech checks with both WebLLM WAV input
and native PCM input returned incorrect text or hit the token limit. The native
browser audio examples remain available for inspecting actual model output;
accepting audio bytes does not establish transcription accuracy.

The current WebLLM Gemma artifact supports text and audio, not images. A real PNG
request fails with `UnsupportedContentPartError`. Its
[manifest](https://huggingface.co/mlc-ai/gemma-4-E2B-it-q4f16_1-MLC/blob/main/mlc-model-manifest.json)
declares no image input, and MLC's
[Gemma implementation](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/model/gemma4/gemma4_model.py)
contains text and audio support. Harness's image reader cannot supply the missing
vision encoder. Other browser media and native APIs have
[separate examples](../browser/README.md).

[imageText](./ocr.ts) uses Tesseract.js to read English and Korean text in PNG
and JPEG files up to 20 MiB. OCR runs on the CPU in this Node.js tool. Its engine
and language files are pinned npm dependencies, loaded locally without a runtime
CDN request. Each call closes its worker and removes its temporary language files
on success, failure or cancellation. Cancellation during initialization is checked
as soon as the worker is ready; recognition can be interrupted immediately.
Image bytes come from the agent's sandbox and are never passed as host paths or
URLs to OCR. Images with no detected text return a tool error. Small text, low
contrast, handwriting and layouts can produce incorrect text.

The tool returns actual OCR text to the agent; the adapter carries it as a normal
Strands tool result. These dependencies belong to the example, not the
`strands-webllm` package. This verifies the adapter's tool loop, thinking and
concurrent request handling. It does not add native image input or general image
understanding to Gemma. Other OCR languages or vision services can be connected
as tools without changing the adapter.

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
import { BrowserModel } from 'strands-webllm'

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
  builtinTools: { read: false, web_fetch: { model: auxModel }, web_search: 'exa' },
  tools: [createRead()],
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

The [recovery test](./test/recovery.browser.ts) destroys the browser page while a
native generation is active, creates a new engine, and resumes its WebLLM
checkpoint. It also creates a new Harness agent with the same session ID and
checks that reasoning, completed tool calls, and their results survive unchanged.
The next answer uses the saved result without running the completed tool again.

These are separate persistence APIs. A Harness session ID restores the conversation;
WebLLM's generation session ID restores native generation. Native checkpoints do
not restore an Agent turn or decide whether an interrupted tool already changed
external state. Use idempotent tool operations or reconcile that state before retrying.

For a native checkpoint created with `extra_body.resumable`, use the original API:

```ts
const result = await page.evaluate(
  (sessionId) =>
    window.webllmBridge.withEngine((engine) =>
      engine.resumeChatCompletion(sessionId, { continueGeneration: true }),
    ),
  generationSessionId,
)
```

Keep the browser's storage and origin, reload the same model, and retain the
checkpoint ID. In WebLLM 0.2.86, calling `resumeChatCompletion(id)` without
`continueGeneration` reads saved text only. Native recovery can return `text_only`
when generation cannot continue; inspect the returned `recoveryMode`. This API
is available for native chat, not Gemma's raw tool/thinking protocol.

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
WebLLM keeps its own model cache and inference state. `web_search: 'exa'` uses
Harness's built-in [Exa MCP integration](https://exa.ai/docs/get-started/exa-mcp).
Search queries leave your environment. The public endpoint works without a key
but has rate limits; `EXA_API_KEY` is optional. Model inference and file reading
remain local. E2B can still choose the wrong source from real search results;
inspect the tool result when an exact URL matters.

Run the real GPU checks with:

```sh
npm run test:browser -- examples/harness/test
```

Tests create isolated temporary files, sessions, skills, and memory. Some use
named choice to test a feature separately from E2B's tool-selection accuracy.
They check actual tool results and final text; no hook supplies an answer.
The older Hermes check remains separate and ends after its tool batch.
Target versions are Harness 0.3.0, Strands SDK 1.21.0, and WebLLM 0.2.86.
