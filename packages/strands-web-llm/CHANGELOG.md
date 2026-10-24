# Changelog

## 0.2.0 — 2026-10-24

- Preserve caller cancellation reasons without reclassifying them as model errors.
- Exclude discarded Qwen reasoning from token estimates.
- Handle nested type exclusions in Qwen3.5 tool schemas.
- Update Strands SDK to 1.22.0 and the Harness example to 0.3.1.
- Support Strands SDK 1.18.0 and newer compatible 1.x releases, plus WebLLM 0.2.79 and newer compatible patch releases.
- Preserve deeply nested Gemma tool values without overflowing the writer.
- Avoid duplicate native tool calls in deeply nested history.
- Support Qwen templates with prefilled thinking through `thinkingPrefilled`.
- Allow `request.extra_body.preserve_thinking` for Gemma tool history.
- Allow required tool choice with one tool on Gemma and native tool models.
- Recognize other WebLLM precisions of the listed Qwen models.
- Preserve tool argument types, literal protocol tags, and interrupted output.
- Match Qwen3.5 argument types to tool schemas with unset optional fields.
- Keep Gemma message IDs and metadata when trimming past thinking.
- Reject malformed output, duplicate call IDs, ambiguous Qwen3.5 argument names and string history, and content after a stop.
- Reject empty model IDs and framework-owned request fields before inference.
- Preserve browser error types and JSON causes.
- Report native WebLLM latency in Strands metrics.
- Expose native log probabilities on Strands stop events.
- Read model identity without copying the full configuration.
- Reduce CPU work and browser reads for streaming, tool JSON, history, and audio.
- Reuse vectors for duplicate document writes through Strands storage indexing.
- Skip invalid UTF-8 entries instead of embedding replacement text.

## 0.1.1

- Preserve input text, empty turns, output block order, and thinking across Harness tool loops.
- Normalize bare JSON mode and match native Gemma, Qwen, and Hermes tool history and output.
- Avoid duplicate Hermes calls after JSON keys are reordered.
- Validate tool restrictions, preserve split names, and reject changed call IDs while streaming.
- Preserve Qwen3.5 argument names and types through enums, constants, patterns, JSON Pointers, and schema unions.
- Avoid repeated expansion of shared Qwen3.5 tool schemas.
- Preserve empty system prompts; allow WebLLM to supply native Hermes tool instructions.
- Keep reserved keys in Gemma array item schemas.
- Preserve protocol text and interrupted output; reject malformed calls, invalid JSON, and signed history.
- Keep tool-result image groups without added narration; model support remains unverified.
- Snapshot queued inputs; preserve JSON keys, error causes, and usage across browser transport.
- Stop callbacks and pending output on cancellation; close streams before reusing the engine.
- Reduce media serialization, browser transport, and embedding search overhead.
- Avoid repeated JSON serialization in mixed tool results.
- Deliver native browser chunks while tool arguments are still being generated.
- Bound embedding batches and storage reads; isolate caches, copy reusable vectors, and reject mismatched dimensions before caching.
- Reuse cached text across copies, renames, and content swaps during search.
- Let FileMemoryStore omit unused document bytes.
- Share engine scheduling and runtime assets across adapters.
- Update examples to verified model support and model-card sampling settings.

With thinking on, Qwen3 0.6B can miscopy arguments or omit answers;
Qwen3 4B can call dependent tools together. Both failures also occur in native WebLLM.

## 0.1.0

- Connect WebLLM models to Strands SDK and Harness.
