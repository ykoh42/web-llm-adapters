# Releases

Update the package version and changelog, then push to `main`.
Publish a GitHub release named `<package>-v<version>`; for example, `strands-web-llm-v0.2.2`.
The Publish workflow runs CI and installation checks, then publishes that package to npm with provenance.
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
