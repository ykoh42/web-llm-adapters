# Changelog

## 0.1.1 (unreleased)

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
