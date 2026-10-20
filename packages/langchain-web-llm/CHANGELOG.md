# Changelog

## Unreleased

- Forward browser `onFailedAttempt` handlers, including async decisions, cancellation, and error updates, without changing model input or output.
- Honor `disableStreaming` and preserve browser embedding retry policies and error metadata.
- Release local failed-attempt callbacks on cancellation, including zero-retry calls.

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
