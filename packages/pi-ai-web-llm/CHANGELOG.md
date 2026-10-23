# Changelog

## 0.2.0 — 2026-10-24

- Handle nested type exclusions in Qwen3.5 tool schemas.
- Update Pi to 1.4.0 and use its official IndexedDB storage in the Durable example.
- Support Pi AI 1.3.0 and newer compatible 1.x releases, plus WebLLM 0.2.85 and newer compatible patch releases.
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
