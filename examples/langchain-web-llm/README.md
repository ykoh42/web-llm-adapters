# LangChain examples

- [Browser](browser/src/main.ts): LangChain chat, thinking, and tool calls with a local WebLLM engine.
- [Deep Agents](deepagents/README.md): a Node.js agent connected to a private browser page.

These are local demos, not a requirement for using the package.

From the repository root, install dependencies and start the local browser example:

```sh
pnpm install
pnpm dev:langchain
```

Open the local URL printed by the server, load a model, and send a message. Web Worker is on by default to keep inference off the UI thread. The first load fetches WebLLM's runtime and model files; later runs use the browser cache.
