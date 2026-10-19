# Changelog

## 0.1.1 (unreleased)

- Use Strands media serialization for faster image and WAV input.
- Reject required Qwen tool choices before inference when no tools are registered.
- Preserve each tool-result image group as a separate user turn; model support remains unverified.
- Snapshot queued document bytes and emit usage metadata before message stop.
- Keep embedding caches separate across stores; bound storage reads and document embedding batches.
- Let FileMemoryStore omit unused bytes from search results.
- Keep cosine scores stable for very large and very small embeddings.
- Drop cached vectors when documents become empty.
- Reject non-object tool arguments before execution.
- Reject signed tool-call history that WebLLM cannot preserve.
- Reduce browser transport and embedding search overhead.
- Preserve SDK model error types across browser calls.
- Preserve literal tool tags inside Qwen JSON arguments.
- Reject overflowing tool numbers in native and model-specific formats.
- Share protocol parsers, engine scheduling, and runtime assets with other adapters.
- Reject tool output that violates an explicit tool choice.
- Preserve Gemma reasoning when a caller-supplied stop string ends generation.
- Match Qwen tool history, reasoning, and Qwen3.5 tool declarations to their templates.
- Accept Qwen3.5 parameter boundaries and extra closing tags handled by its recommended parser.
- Sort Gemma object keys to match its template.
- Use model-card sampling settings in examples and update verified model support.

With thinking on, Qwen3 0.6B can miscopy arguments or omit answers;
Qwen3 4B can call dependent tools together. Both failures also occur in native WebLLM.

## 0.1.0

- Connect WebLLM models to Strands SDK and Harness.
