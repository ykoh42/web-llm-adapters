"""Rebuild the model library from pinned sources. No model weights are needed."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import ssl
import subprocess
import sys
import tarfile
from urllib.request import urlopen

import certifi


runtime = Path(__file__).resolve().parent
manifest = json.loads((runtime / "manifest.json").read_text())


def run(*args, cwd=None, env=None):
    subprocess.run([str(arg) for arg in args], cwd=cwd, env=env, check=True)


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def fetch(source, archive, destination):
    if not archive.exists():
        context = ssl.create_default_context(cafile=certifi.where())
        partial = archive.with_suffix(".part")
        with urlopen(source["url"], timeout=120, context=context) as response, partial.open("wb") as file:
            shutil.copyfileobj(response, file)
        partial.replace(archive)
    if digest(archive) != source["sha256"]:
        raise ValueError(f"Source checksum mismatch: {archive}")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as file:
        members = file.getmembers()
        for member in members:
            member.name = member.name.partition("/")[2]
        file.extractall(destination, members=[m for m in members if m.name], filter="data")


def build(work, output, jobs):
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use Python 3.12.")
    llvm = shutil.which("llvm-config")
    emcc = shutil.which("emcc")
    for command, version in [(llvm, "22.1.8"), (emcc, "6.0.12")]:
        if not command:
            raise RuntimeError("Put LLVM 22.1.8 and Emscripten 6.0.12 on PATH.")
        actual = subprocess.check_output([command, "--version"], text=True)
        if version not in actual.splitlines()[0]:
            raise RuntimeError(f"Expected {version}: {actual.splitlines()[0]}")
    for name in ["cache.patch", "projection.patch", "model.json"]:
        if digest(runtime / name) != manifest["files"][name]:
            raise ValueError(f"Build input checksum mismatch: {name}")

    downloads = work / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    for name, source in manifest["sources"].items():
        fetch(source, downloads / (name + ".tar.gz"), work / source["path"])
    tvm, mlc = work / "tvm", work / "mlc"
    run("patch", "-p1", "-i", runtime / "cache.patch", cwd=tvm)
    run("patch", "-p1", "-i", runtime / "projection.patch", cwd=mlc)

    prefix_map = f"-ffile-prefix-map={work}=/src"
    env = {
        **os.environ,
        # Archives have no Git metadata; do not embed the enclosing adapter's revision.
        "GIT_CEILING_DIRECTORIES": str(work),
        "SETUPTOOLS_SCM_PRETEND_VERSION": "0.1.0",
        "CMAKE_ARGS": "-DTVM_FFI_USE_LIBBACKTRACE=OFF",
        "EMCC_CFLAGS": shlex.join(["-sDEFAULT_TO_CXX=1", prefix_map]),
    }
    run(
        sys.executable, "-m", "pip", "install", "--no-build-isolation",
        tvm / "3rdparty/tvm-ffi", env=env,
    )
    run(
        "cmake", "-S", tvm, "-B", tvm / "build", "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", f"-DUSE_LLVM={llvm}",
        "-DUSE_LIBBACKTRACE=OFF", "-DTVM_FFI_USE_LIBBACKTRACE=OFF", env=env,
    )
    run("cmake", "--build", tvm / "build", "--parallel", jobs, env=env)
    run(
        os.environ.get("CXX", "c++"), "-std=c++17", runtime / "check-cache.cc",
        "-I" + str(tvm / "include"), "-I" + str(tvm / "3rdparty/tvm-ffi/include"),
        "-I" + str(tvm / "3rdparty/tvm-ffi/3rdparty/dlpack/include"),
        "-I" + str(tvm / "src/runtime/vm"), "-L" + str(tvm / "build/lib"),
        "-ltvm_runtime", "-ltvm_ffi", "-Wl,-rpath," + str(tvm / "build/lib"),
        "-o", work / "check-cache", env=env,
    )
    run(work / "check-cache", env=env)
    # Make does not track flag changes; rebuild every distributed runtime object.
    for path in [
        tvm / "web/dist/wasm/wasm_runtime.bc",
        tvm / "web/dist/wasm/tvmjs_support.bc",
        tvm / "web/dist/wasm/webgpu_runtime.bc",
        mlc / "web/dist/wasm/mlc_wasm_runtime.bc",
    ]:
        path.unlink(missing_ok=True)
    # The upstream Makefiles replace EMCC_CFLAGS, so pass the map to EMCC too.
    compiler = shlex.join([emcc, prefix_map])
    run(
        "make", "-C", tvm / "web", f"EMCC={compiler}", "-j", jobs,
        "dist/wasm/wasm_runtime.bc", "dist/wasm/tvmjs_support.bc",
        "dist/wasm/webgpu_runtime.bc", env=env,
    )
    run(
        "make", "-C", mlc / "web", f"EMCC={compiler}", f"TVM_SOURCE_DIR={tvm}",
        "dist/wasm/mlc_wasm_runtime.bc", env=env,
    )
    env.update({
        "TVM_HOME": str(tvm),
        "TVM_SOURCE_DIR": str(tvm),
        "TVM_LIBRARY_PATH": str(tvm / "build/lib"),
        "MLC_LLM_SOURCE_DIR": str(mlc),
        "MLC_LIBRARY_PATH": str(mlc / "web/dist/wasm"),
        "SKIP_LOADING_MLCLLM_SO": "1",
        "PYTHONPATH": os.pathsep.join([str(tvm / "python"), str(mlc / "python")]),
        "TVM_NUM_THREADS": str(jobs),
        "BINARYEN_CORES": str(jobs),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        sys.executable, "-m", "mlc_llm", "compile", runtime / "model.json",
        "--quantization", "q4f16_1", "--device", "webgpu",
        "--overrides", "prefill_chunk_size=512", "--output", output, env=env,
    )
    output.chmod(0o644)
    record = {"output": str(output), "sha256": digest(output), "inputs": manifest}
    output.with_suffix(".build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Built {output}; run the browser checks before replacing the packaged file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path(".cache/runtime-build"))
    parser.add_argument("--output", type=Path, default=Path(".cache/runtime-build/gemma-4-e2b.wasm"))
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    build(args.work.resolve(), args.output.resolve(), args.jobs)
