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

| Example                | Read                       | What it does                                         |
| ---------------------- | -------------------------- | ---------------------------------------------------- |
| Chat                   | [chat.ts](src/chat.ts)     | Strands text streaming, usage, and cancellation      |
| JSON schema + logprobs | [chat.ts](src/chat.ts)     | JSON schema, seed, and native token probabilities    |
| Grammar                | [chat.ts](src/chat.ts)     | Restrict output to three colors with EBNF            |
| Tool call              | [tools.ts](src/tools.ts)   | Hermes calls a local Strands clock tool              |
| Image / WAV audio      | [media.ts](src/media.ts)   | Pass file bytes through Strands content blocks       |
| Native PCM audio       | [native.ts](src/native.ts) | Decode audio, mix to mono, and pass Float32 samples  |
| Native embeddings      | [native.ts](src/native.ts) | Use a dedicated embedding model                      |
| Native text completion | [native.ts](src/native.ts) | Raw prompt and two non-streaming choices             |
| Native save + resume   | [native.ts](src/native.ts) | Interrupt after eight chunks, recover, then continue |

[engines.ts](src/engines.ts) owns loading and unloading. A Web Worker is terminated
when unloaded. The Service Worker handler stays registered; unloading still frees
the model. Its startup handler is in [service-worker.ts](src/service-worker.ts).
[output.ts](src/output.ts) only reads Strands events. Feature files call WebLLM and
Strands directly; they do not depend on the page's DOM.

Gemma 4 E2B handles text and audio in WebLLM 0.2.86. It does not accept native tools
or images in this release. The tool example selects Hermes, and the image example
selects Phi vision. These models require separate, large downloads. For tool mode,
WebLLM adds its own system prompt and JSON grammar. The example explicitly ends the
turn after the tools execute so that Hermes is not asked for another tool batch.

**Details** shows usage, native metadata, and tool results. **Check cache** and
**Delete cached model** use the selected cache backend. Deletion is only enabled
after unloading. OPFS and IndexedDB have separate stored copies. The experimental
cross-origin extension backend is available through native WebLLM configuration;
this page does not install or configure an extension.

The resume example demonstrates an orderly interruption, not a browser crash.
It deletes its session after a successful continuation. A manually stopped or
failed run can leave a native persisted session; inspect or remove it through
`engine.listResumableSessions()` and `engine.deleteResumableSession(id)`.
This persistence does not restore Strands messages or tool state.

Run the real model checks with `npm run test:browser`. They need Chrome, WebGPU,
several GB of free disk space, and enough GPU memory. See the [support table](../../docs/support.md)
for the precise verification scope.
