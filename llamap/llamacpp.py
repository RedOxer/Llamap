"""Calls to the tools of the user's llama.cpp build.

The llama.cpp directory comes from --llama-cpp or LLAMA_CPP_DIR. It must be built:
build/bin/llama-quantize, llama-imatrix and llama-perplexity, and
convert_hf_to_gguf.py at the root.
"""
import os
import subprocess
import sys
from pathlib import Path


class LlamaCpp:
    def __init__(self, root=None):
        root = root or os.environ.get("LLAMA_CPP_DIR")
        if not root:
            raise ValueError("llama.cpp directory not set: use --llama-cpp or LLAMA_CPP_DIR")
        self.root = Path(root).expanduser()
        self.bin = self.root / "build" / "bin"
        missing = [t for t in ("llama-quantize", "llama-imatrix", "llama-perplexity") if not (self.bin / t).exists()]
        if missing or not (self.root / "convert_hf_to_gguf.py").exists():
            raise FileNotFoundError(f"incomplete llama.cpp build in {self.root}: {missing or 'convert_hf_to_gguf.py'}")

    def _run(self, cmd):
        r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{Path(cmd[0]).name} failed:\n{r.stderr[-2000:]}")
        return r.stdout

    def tool(self, name, *args):
        return self._run([self.bin / name, *args])

    def convert(self, hf_dir, out):
        """HF model -> BF16 GGUF."""
        self._run([sys.executable, self.root / "convert_hf_to_gguf.py", hf_dir, "--outtype", "bf16", "--outfile", out])

    def imatrix(self, model, text, out, chunks=200, ctx=512):
        self.tool("llama-imatrix", "-m", model, "-f", text, "-c", ctx, "--chunks", chunks, "-ngl", 99, "-o", out)

    def quantize(self, src, out, preset, imatrix, *, pure=False, tensor_types=None, token_embd=None, output=None):
        args = ["--imatrix", imatrix]
        if pure:
            args.append("--pure")
        if tensor_types:
            args += ["--tensor-type-file", tensor_types]
        if token_embd:
            args += ["--token-embedding-type", token_embd]
        if output:
            args += ["--output-tensor-type", output]
        self.tool("llama-quantize", *args, src, out, preset)

    def ggml(self, threads=1):
        from llamap.ggml import Ggml
        return Ggml(self.bin / "libggml-base.so", threads)
