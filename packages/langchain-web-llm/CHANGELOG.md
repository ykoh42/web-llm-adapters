# Changelog

## Unreleased

- Reject detached image and audio bytes instead of sending empty data.
- Remove WebLLM's empty thinking prefix from Qwen2.5 answers.
- Accept uppercase data URL schemes for WAV audio.
- Preserve literal Qwen thinking tags in answers when thinking is disabled.
- Cancel requests promptly while WebLLM or browser responses are pending.
- Return promptly when closing a cancelled stream; keep the engine reserved for cleanup.
- Skip queued retry callbacks after cancellation.
- Wait for native cleanup when closing browser bridges.

- Preserve negative zero in tool calls and conversation history.
- Preserve error fields when a provider rejects with an object.
- Preserve context overflow causes.
- Support `require()` in Node.js CommonJS.
- Stop scheduling retries after cancellation with a custom reason.
- Stop requests cancelled during payload preparation before opening WebLLM.
- Stream tool arguments and expose call IDs only after validation.
- Preserve reserved keys in streamed tool calls from browser models.
- Use content events supported by LangChain Core 1.2.8 and later.
- Keep text, thinking, and tool events in separate blocks.
- Reject duplicate tool call IDs before returning tool calls.
- Accept Qwen3.5 string enums and string `const` values without an explicit type.
- Constrain type-less Qwen3.5 non-string `enum`/`const` values and JSON-only unions.
- Allow explicit text format with constrained Qwen tool calls.
- Reject incomplete or malformed Qwen tool calls instead of repairing their output.
- Preserve cross-realm error types, causes, and provider fields across browser bridges.
- Accept image, WAV, and PCM arrays created in another JavaScript realm.
- Await async `onChunk` handlers and propagate their errors.
- Keep browser errors and retries when provider fields cannot be inspected.
- Preserve native response creation times in metadata.
- Report both WebLLM Phi-3.5 Vision precisions in model profiles.
- Reject Qwen3.5 tool names that cannot be represented in XML.
- Honor token callbacks enabled during a request.
- Keep `preserve_thinking` when call-level requests set WebLLM options.
- Reject non-array document lists before embedding.
- Exclude runtime build tools from the npm package.
- Avoid repeated Qwen tool-format serialization when no payload hook is used.

## 0.3.0 — 2026-10-25

- Align example sampling and Qwen3.5 engine setup with the model cards.
- Remove WebLLM’s fixed thinking-off prefix from Qwen JSON and grammar output; preserve the body and raw chunks.
- Preserve empty Qwen3.5 thinking boundaries in current tool history.
- Keep malformed or interrupted Qwen tool syntax out of answer text; retain raw output through `onChunk`.
- Accept Qwen3.5 boolean spellings handled by its recommended parser.
- Add opt-in Qwen tool-call constraints with required selection, thinking, and Qwen3.5 XML support through WebLLM 0.2.86.

## 0.2.1 — 2026-10-24

- Allow Qwen3 1.7B tool calls instead of blocking requests based on model output quality.

## 0.2.0 — 2026-10-24

- Handle nested type exclusions in Qwen3.5 tool schemas.
- Preserve non-Error callback failures as causes in browser models.
- Match Qwen3.5 argument types to tool schemas with unset optional fields.
- Accept `ChatMessage` with user, assistant, system, and LangChain human/ai roles.
- Support LangChain Core 1.2.8 and newer compatible 1.x releases.
- Support WebLLM 0.2.79 and newer compatible patch releases.
- Report unsupported mid-conversation system and tool capabilities.
- Preserve deeply nested Gemma tool values without overflowing the writer.
- Avoid duplicate native tool calls in deeply nested history.
- Support Qwen templates with prefilled thinking through `thinkingPrefilled`.
- Allow `strict: true` with native `jsonSchema` output.
- Allow `request.extra_body.preserve_thinking` for Gemma tool history.
- Allow required tool choice with one tool on Gemma and native tool models.
- Recognize other WebLLM precisions of the listed Qwen models.
- Fix structured output selection for Gemma thinking and models without tools.
- Preserve tool argument types, literal protocol tags, and interrupted output.
- Reject malformed output, duplicate call IDs, and ambiguous Qwen3.5 argument names and string history.
- Preserve native response IDs, model IDs, and token probabilities.
- Support Float32 PCM, inline WAV URLs, and standard WAV MIME names.
- Honor retry settings, callbacks, cancellation, and `disableStreaming` in local and browser models.
- Preserve browser error types, JSON causes, and failed-attempt callback edits.
- Add `withEngine` to the browser bridge for native API calls.
- Keep response-cache entries separate by tool definitions and output format.
- Report native Hermes media limits to Deep Agents.
- Reject framework-owned request fields and content received after the stop reason.
- Reduce CPU work and browser reads for streaming, tool JSON, history, and audio.

## 0.1.0

- Add WebLLM chat, tools, thinking, and embeddings for LangChain, LangGraph, and Deep Agents.
- Keep Node.js tools and private artifacts local; stream browser responses through standard callbacks.
- Deliver native browser chunks while tool arguments are still being generated.
- Reduce image and WAV byte encoding overhead.
- Preserve input text, JSON keys, media bytes, and output order without filler.
- Normalize bare JSON mode and model protocols; preserve interrupted output and reject malformed calls and invalid JSON.
- Avoid duplicate Hermes calls after JSON keys are reordered.
- Reject changed tool call IDs while streaming.
- Preserve Qwen3.5 argument names and types through enums, constants, patterns, JSON Pointers, and schema unions.
- Avoid repeated expansion of shared Qwen3.5 tool schemas.
- Preserve empty system prompts; allow WebLLM to supply native Hermes tool instructions.
- Keep reserved keys in Gemma array item schemas.
- Preserve response schemas across sampling overrides; forward per-call retries and native metadata.
- Type usage metadata for invoke, stream, and bound tools under strict TypeScript settings.
- Keep queued and browser inputs stable, tool calls in native order, and error types and causes intact.
- Stop callbacks and pending output on cancellation; close streams before reusing the engine.
- Expose context-limit errors as LangChain's non-retryable `ContextOverflowError` for agent recovery.
- Support image URLs, image/WAV bytes, and custom profiles; reject media forms WebLLM cannot preserve.
- Retry only failed embedding batches; reject malformed vectors without retrying them.
- Keep embedding results stable when the engine reuses its output arrays.
- Accept whitespace embedding inputs without changing the text; reject invalid native vectors.
- Keep examples within the selected model's context window.
