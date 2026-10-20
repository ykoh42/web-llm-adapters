# Gemma 4 E2B runtime

A bundled fix for WebLLM **0.2.86**, for `gemma-4-E2B-it-q4f16_1-MLC` only.
It fixes cache corruption and projection rounding while keeping the original model weights.
Included in both `strands-web-llm` and `langchain-web-llm`; no local compiler is needed.

## Use

For Vite, import the asset URL and its pinned integrity hash. For LangChain,
use `langchain-web-llm` in both asset imports:

```ts
import { CreateMLCEngine, prebuiltAppConfig } from '@mlc-ai/web-llm'
import runtimeUrl from 'strands-web-llm/runtime/gemma-4-e2b.wasm?url'
import runtime from 'strands-web-llm/runtime/manifest.json'

const appConfig = {
  ...prebuiltAppConfig,
  model_list: prebuiltAppConfig.model_list.map((model) =>
    model.model_id === runtime.model_id
      ? {
          ...model,
          model_lib: new URL(runtimeUrl, window.location.href).href,
          integrity: { ...model.integrity, model_lib: runtime.integrity },
        }
      : model,
  ),
}
const engine = await CreateMLCEngine(
  runtime.model_id,
  { appConfig },
  {
    context_window_size: 16384,
  },
)
```

Pass this engine to your adapter as usual. Workers
accept the same `appConfig`. The repository's browser and Harness examples
already use this setup. The provider itself leaves engine setup to the caller.

Without an asset bundler, copy the exported Wasm file to your site's public
directory and use its absolute URL as `model_lib`. Serve it as `application/wasm`;
cross-origin hosting also needs CORS. Keep the supplied integrity value.

## Protocol limits

Gemma tools and thinking use its native prompt template through raw completions.
That mode accepts text and tool results, but cannot combine media, constrained
output, native generation recovery, or `ignore_eos`. WebLLM re-prefills raw prompts
on each turn. Ordinary chat uses WebLLM's chat API and KV cache.
Past thoughts remain in agent history and leave the next user turn's prompt.

<details>
<summary>Rebuild and implementation details (maintainers)</summary>

Sliding-attention and full-attention metadata use separate buffers. Projections
accumulate in float32 before returning to float16 activations. Local M4 Pro checks
measured about 0.8% more input processing time and unchanged generation speed;
results on other devices may differ.

Rebuilding is for maintainers. Use Python 3.12, LLVM **22.1.8** (with `llvm-config`
on `PATH`), Emscripten **6.0.12** (`emsdk_env.sh` sourced), and a C++ toolchain.
From the repository root:

```sh
python3.12 -m venv .cache/runtime-venv
.cache/runtime-venv/bin/python -m pip install -r shared/runtime/requirements.txt
.cache/runtime-venv/bin/python shared/runtime/build.py
```

The script verifies pinned source archives and build inputs, applies
[cache.patch](cache.patch) and [projection.patch](projection.patch), checks that all four cache metadata arrays stay
separate at both depths, and builds under `.cache/runtime-build`. It downloads
compiler sources, not model weights. The output has a `.build.json` record;
compiler output can make rebuilt bytes differ. The shipped file's hash is in
[manifest.json](manifest.json).

Before replacing a runtime, verify its cache behavior and run the real Harness
tests. Ordinary CI checks hashes and package contents without using a GPU.

</details>
