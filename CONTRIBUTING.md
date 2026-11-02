# Contributing

Use the Node.js version in `.node-version` and pnpm 12.16.1. Install dependencies and run the checks:

```sh
npm install --global pnpm@12.16.1
pnpm install
pnpm check
```

Keep changes focused on the affected adapter. Put shared code in `shared/` only when adapters need the same behavior. Run `pnpm test:install` for package or dependency changes. For model or protocol changes, follow [release verification](.github/RELEASING.md).

Check existing issues and pull requests before opening a new one. Pull requests should describe the behavior changed and the checks run. See [AGENTS.md](AGENTS.md) for code guidelines.
