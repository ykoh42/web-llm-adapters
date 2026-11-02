# Contributing

Use the Node.js version in `.node-version` and npm 11.21.1:

```sh
npm ci
npm run check
```

Keep changes focused on the affected adapter. Put shared code in `shared/` only when adapters need the same behavior. Run `npm run test:install` for package or dependency changes. For model or protocol changes, follow [release verification](.github/RELEASING.md).

Check existing issues and pull requests before opening a new one. Pull requests should describe the behavior changed and the checks run. See [AGENTS.md](AGENTS.md) for code guidelines.
