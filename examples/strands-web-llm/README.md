# Strands examples

Examples and integration tests for `strands-web-llm`.

| Example                      | Purpose                                                |
| ---------------------------- | ------------------------------------------------------ |
| [Browser](browser/README.md) | Strands SDK chat, tools, media, and native WebLLM APIs |
| [Harness](harness/README.md) | Node.js tools with inference in a browser              |

The browser app is a local demo and test page. It is not required to use the package.
Each example keeps its own tests. [fixtures/browser.ts](fixtures/browser.ts) handles shared browser setup and cleanup.
Gemma 4 E2B is one example model. Available features depend on the WebLLM model you load.
