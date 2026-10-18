# Changelog

## 0.1.1

- Match Qwen tool history, reasoning, and Qwen3.5 tool declarations to their templates.
- Accept Qwen3.5 parameter boundaries and extra closing tags handled by its recommended parser.
- Sort Gemma object keys to match its template.
- Use model-card sampling settings in examples and update verified model support.

With thinking on, Qwen3 0.6B can miscopy arguments or omit answers;
Qwen3 4B can call dependent tools together. Both failures also occur in native WebLLM.

## 0.1.0

- Connect WebLLM models to Strands SDK and Harness.
