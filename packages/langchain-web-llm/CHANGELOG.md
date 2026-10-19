# Changelog

## 0.1.0 (unreleased)

- Run token callbacks before yielding stream chunks.
- Forward reasoning, tool calls, and metadata through standard callbacks and agent streams.
- Forward agent JSON schemas and per-call retry settings to WebLLM.
- Map tool-result images to user turns and let custom profiles declare model support.
- Keep Deep Agents tool images when custom profiles support vision.
- Accept byte-backed image and WAV content blocks.
- Bound embedding request batches with a configurable `batchSize`.
- Reject required Qwen tool choices before inference when no tools are registered.
- Match the browser example's input limit to WebLLM's selected model config.
- Keep reasoning and tool call indices distinct in LangChain content streams.
- Reject unsupported image detail and text/media order that cannot be preserved.
- Keep overlapping browser stream reads in order without duplicate chunks.
- Preserve literal tool tags inside Qwen JSON arguments.
- Reject overflowing tool numbers in native and model-specific formats.
- Add WebLLM chat, tools, reasoning, and embeddings for LangChain and LangGraph.
- Add a browser bridge for Node.js agents, including Deep Agents.
- Share model protocol parsers and engine scheduling with the Strands adapter.
