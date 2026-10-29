# Releases

Update the package version and changelog, then commit the changes.
Run `pnpm test:release:models` locally with Chrome and WebGPU. It reuses the tagged real-model tests for all three adapters.
For performance-sensitive changes, compare the affected adapter with native WebLLM using the [LangChain](../examples/langchain-web-llm/browser/test/browser/performance.spec.ts), [Pi](../examples/pi-ai-web-llm/browser/test/performance.spec.ts), or [Strands](../examples/strands-web-llm/harness/test/browser/performance.browser.ts) performance test; these are not part of the tagged model checks.
A skipped comparison does not verify performance.
The tests also check that each adapter can resume interrupted text generation through `withEngine()`.
Thinking checks require reasoning before a tool call and preservation in the next request; skipped reasoning leaves that check unverified.
`O*` tool support requires a constrained tool loop and a separate default-generation protocol check. Native malformed output must raise the framework's error without executing that call; it does not count as a successful tool loop.
Commit the generated `.github/real-models.json`, run `pnpm check:release:models`, then push to `main`.
Publish a GitHub release named `<package>-v<version>`; for example, `strands-web-llm-v0.2.2`.
The Publish workflow checks CI, packed installs, and the real-model report before publishing to npm with provenance.
Code, dependency, runtime, or support-table changes need matching results. Skipped, failed, and missing checks block publishing; documentation-only changes can reuse results.
GPU tests run locally, with no paid CI runner.
Use **Run workflow** to verify an existing tag without publishing.

## One-time npm setup

Register `.github/workflows/publish.yml` for each package using npm's [Trusted Publishing](https://docs.npmjs.com/trusted-publishers/):

```sh
for package in strands-web-llm langchain-web-llm pi-ai-web-llm; do
  npm trust github "$package" --repo ykoh42/web-llm-adapters --file publish.yml --allow-publish --yes
  sleep 2
done
```

No npm token is stored in GitHub.
