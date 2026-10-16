# Browser examples

From the repository root:

```sh
npm ci
npm run dev
```

Open <http://127.0.0.1:4173>. Choose an **Example** and **Engine**, click **Load model**,
then **Send**. **Unload** before changing either. Each run starts a new conversation.
The model list comes from the installed WebLLM catalog; choosing an incompatible
model reports WebLLM's error. FP16 models need `shader-f16`. FP32 variants use more memory.

| Example                   | Read                           | What it does                                                   |
| ------------------------- | ------------------------------ | -------------------------------------------------------------- |
| Chat                      | [chat.ts](src/chat.ts)         | Strands text streaming, usage, and cancellation                |
| JSON schema + logprobs    | [chat.ts](src/chat.ts)         | JSON schema, seed, and native token probabilities              |
| Grammar                   | [chat.ts](src/chat.ts)         | Restrict output to three colors with EBNF                      |
| Structural tags           | [advanced.ts](src/advanced.ts) | Constrain tagged JSON                                          |
| Qwen thinking on and off  | [advanced.ts](src/advanced.ts) | Compare empty and populated thinking blocks                    |
| Gemma thinking on and off | [advanced.ts](src/advanced.ts) | Compare thinking modes; reasoning stays in Details             |
| Gemma thinking + tools    | [tools.ts](src/tools.ts)       | Think, call the clock, then answer using the result            |
| Sampling + latency        | [advanced.ts](src/advanced.ts) | Repeat a seeded sample, inspect timings, and stop at a newline |
| Gemma tool call + answer  | [tools.ts](src/tools.ts)       | Gemma calls a clock tool, then answers with the result         |
| Hermes tool batch         | [tools.ts](src/tools.ts)       | Legacy native tool API, ends after executing the tools         |
| Image / WAV audio         | [media.ts](src/media.ts)       | Pass file bytes through Strands content blocks                 |
| Native PCM audio          | [native.ts](src/native.ts)     | Decode audio, mix to mono, and pass Float32 samples            |
| Native embeddings         | [native.ts](src/native.ts)     | Use a dedicated embedding model                                |
| Native text completion    | [native.ts](src/native.ts)     | Raw prompt and two non-streaming choices                       |
| Native save + resume      | [native.ts](src/native.ts)     | Interrupt after eight chunks, recover, then continue           |

[engines.ts](src/engines.ts) owns loading and unloading. A Web Worker is terminated
when unloaded. The Service Worker handler stays registered; unloading still frees
the model. Its startup handler is in [service-worker.ts](src/service-worker.ts).
[output.ts](src/output.ts) only reads Strands events. Feature files call WebLLM and
Strands directly; they do not depend on the page's DOM.

Gemma 4 E2B handles text and audio in WebLLM 0.2.86. The adapter also supports its
native function-call format through raw completions: the clock result returns to
Gemma for a final answer. WebLLM's separate native `tools` API remains restricted
to Hermes; the legacy Hermes example ends after the tool batch. The image example
uses Phi vision. These other models need separate, large downloads. See the
[provider limits](../../packages/strands-webllm/README.md#limits) for tool-mode options.

**Details** shows usage, native metadata, and tool results.
Gemma reasoning is shown separately in **Details**; **Response** contains only
answer text. Raw native chunks still contain the original thinking control tokens.

**Check cache** and
**Delete cached model** use the selected cache backend. Deletion is only enabled
after unloading. OPFS and IndexedDB have separate stored copies. The experimental
cross-origin backend needs the separate Cross-Origin Storage extension and a
localhost page. Choose **Main thread** for that backend. Shared cache deletion is
disabled because the upstream extension has no delete API. Model weights can be
shared across origins; small config and hash lookup requests can still use the network.

The resume example demonstrates an orderly interruption, not a browser crash.
It deletes its session after a successful continuation. A manually stopped or
failed run can leave a native persisted session; inspect or remove it through
`engine.listResumableSessions()` and `engine.deleteResumableSession(id)`.
This persistence does not restore Strands messages or tool state.

Open [engine.html](engine.html) on the same local server for the lower-level examples:

| Example         | Source                                  | What it checks                                                                  |
| --------------- | --------------------------------------- | ------------------------------------------------------------------------------- |
| Multiple models | [models.ts](src/checks/models.ts)       | Two custom chat IDs and an embedding model loaded together                      |
| Logit processor | [processor.ts](src/checks/processor.ts) | Fixed token sampling, callbacks, and reset on main thread and Worker            |
| Engine state    | [state.ts](src/checks/state.ts)         | Progress, GPU info, logging, history, stats, and reset                          |
| Integrity       | [integrity.ts](src/checks/integrity.ts) | Custom manifest model with valid and invalid config, Wasm, and tokenizer hashes |

The integrity example computes hashes to demonstrate validation. A real release
must pin hashes obtained from a trusted publisher. `resetChat()` resets native KV
and conversation state; it does not erase Strands history or WebLLM's last output string.

For the Chrome extension worker, run `npm run build`, open `chrome://extensions`,
enable Developer mode, and load `examples/browser/dist` as an unpacked extension.
Open its `index.html` page and choose **Chrome extension worker**. Its handler is in
[background.ts](src/extension/background.ts). Unload frees the model and disconnects the client port.

`npm run test:browser` checks the ordinary pages with Chrome and real WebGPU.
`npx playwright install chromium` followed by `npm run test:extensions` checks the
real extension worker and cross-origin cache. Both need free disk space and GPU
memory. [Harness has its own runnable example](../harness/README.md).
