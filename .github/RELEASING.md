# Releases

Update the package version and changelog, then commit the changes.
Run `pnpm test:release:models` locally with Chrome and WebGPU. It reuses the tagged real-model tests for all three adapters.
`O*` tool support requires a constrained tool loop and a separate default-generation protocol check. Native malformed output must raise the framework's error without executing that call; it does not count as a successful tool loop.
Commit the generated `.github/real-models.json`, run `pnpm check:release:models`, then push to `main`.
Publish a GitHub release named `<package>-v<version>`; for example, `strands-web-llm-v0.2.2`.
The Publish workflow checks CI, packed installs, and the real-model report before publishing to npm with provenance.
Code, dependency, runtime, or support-table changes need matching results. Skipped, failed, and missing checks block publishing; documentation-only changes can reuse results.
GPU tests run locally, with no paid CI runner.
Use its manual **Run workflow** option with **Check without publishing** enabled to verify an existing tag.

## One-time npm setup

Register `.github/workflows/publish.yml` for each package using npm's [Trusted Publishing](https://docs.npmjs.com/trusted-publishers/):

```sh
for package in strands-web-llm langchain-web-llm pi-ai-web-llm; do
  npm trust github "$package" --repo ykoh42/web-llm-adapters --file publish.yml --allow-publish --yes
  sleep 2
done
```

No npm token is stored in GitHub.
