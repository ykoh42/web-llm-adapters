# Changelog

## Unreleased

- Reject Qwen and Gemma tool settings, and Gemma thinking, added through `onPayload`.
- Remove WebLLM's empty thinking prefix from Qwen2.5 answers.
- Preserve literal Qwen thinking tags in answers when thinking is disabled.
- Cancel requests promptly while WebLLM opens or streams a response.
- Return promptly when closing a cancelled stream; keep the engine reserved for cleanup.
- Preserve negative zero in tool calls and conversation history.
- Honor timeouts and cancellation without `AbortSignal.any()`.
- Support `require()` in Node.js CommonJS.
- Allow a tool-call ID to be reused after its earlier call has a result.
- Reject edits to generated Qwen tool constraints made by `onPayload`.
- Stop requests cancelled during payload preparation before opening WebLLM.
- Skip payload and stream wrappers when their callbacks are absent.
- Stream partial tool arguments and withhold `toolcall_end` for incomplete calls.
- Accept Qwen3.5 string enums and string `const` values without an explicit type.
- Constrain type-less Qwen3.5 non-string `enum`/`const` values and JSON-only unions.
- Allow explicit text format with constrained Qwen tool calls.
- Reject incomplete or malformed Qwen tool calls instead of repairing their output.
- Stream array text without buffering when `onPayload` removes native tools.
- Await async `onChunk` handlers and propagate their errors.
- Detect built-in Phi-3.5 Vision models for Pi image input.
- Classify `AbortError` failures from other realms as cancellations.
- Reject Gemma payload edits that change an open protocol prefix.
- Reject Qwen3.5 tool names that cannot be represented in XML.
- Support WebLLM's named function choice in Pi API calls.
- Keep `preserve_thinking` when Pi sampling parameters set WebLLM options.
- Exclude runtime build tools from the npm package.
- Avoid empty image-buffer allocations while converting text-only turns.

## 0.3.0 — 2026-10-25

- Align example sampling and Qwen3.5 engine setup with the model cards.
- Remove WebLLM’s fixed thinking-off prefix from Qwen JSON and grammar output; preserve the body and raw chunks.
- Preserve empty Qwen3.5 thinking boundaries in current tool history.
- Keep malformed or interrupted Qwen tool syntax out of answer text; retain raw output through `onChunk`.
- Accept Qwen3.5 boolean spellings handled by its recommended parser.
- Add opt-in Qwen tool-call constraints with thinking and Qwen3.5 XML support through WebLLM 0.2.86.

## 0.2.1 — 2026-10-24

- Allow Qwen3 1.7B tool calls instead of blocking requests based on model output quality.

## 0.2.0 — 2026-10-24

- Handle nested type exclusions in Qwen3.5 tool schemas.
- Update Pi to 1.4.0 and use its official IndexedDB storage in the Durable example.
- Support Pi AI 1.0.2 and newer compatible 1.x releases, plus WebLLM 0.2.79 and newer compatible patch releases.
- Preserve deeply nested Gemma tool values without overflowing the writer.
- Avoid duplicate native tool calls in deeply nested history.
- Support Qwen templates with prefilled thinking through `thinkingPrefilled`.
- Allow `request.extra_body.preserve_thinking` for Gemma tool history.
- Recognize other WebLLM precisions of the listed Qwen models.
- Preserve tool argument types, literal protocol tags, and interrupted output.
- Reject malformed output, ambiguous Qwen3.5 argument names and string history, unsupported tool choices, and model identity changes.
- Preserve native response IDs, model IDs, and raw stop reasons.
- Respect Pi context-aware token limits, thinking levels, and sampling precedence.
- Honor response formats and stop strings changed by `onPayload`; normalize bare JSON mode.
- Cancel pending payload and stream callbacks; preserve unreadable errors.
- Keep the engine, model, and observer independent of later options-object edits.
- Add `withEngine` for native API calls.
- Reduce CPU work for streaming, tool JSON, and history.

## 0.1.0 — 2026-10-21

- Add a browser-local Pi AI provider for caller-owned WebLLM engines.
- Translate Pi 1.3 transcripts and text, thinking, tool, usage, cancellation, and error events through shared WebLLM protocols.
- Include a Worker-based Pi Agent Core example and real-model browser checks.
