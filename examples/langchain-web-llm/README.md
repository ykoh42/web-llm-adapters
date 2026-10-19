# LangChain examples

- [Browser](browser/src/main.ts): LangChain chat, thinking, and tool calls with a local WebLLM engine.
- [Deep Agents](deepagents/README.md): a Node.js agent connected to a private browser page.

These are local demos, not a requirement for using the package.

From the repository root, install dependencies and start the local browser example:

```sh
npm ci
npm run dev:langchain
```

Open the local URL printed by the server, load a model, and send a message. Select Worker to keep inference off the UI thread. The first load downloads model files; later runs use the browser cache.
