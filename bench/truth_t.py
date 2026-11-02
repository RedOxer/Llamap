"""Measurement: ground truth for the costs, one degraded tensor at a time.

  base    = the whole model in --base-type (Q8_0), with the imatrix;
  variant = the base, except ONE block matrix in --fmt (Q2_K);
  ΔKL     = KL(variant) - KL(base), KL against the BF16 logits on --text.
This is bartowski's degrade-one protocol. One llama-quantize and one
llama-perplexity run per tensor (~17 s per tensor for a 1.7B on an RTX 5090).
Resumable: tensors already in --out are skipped.

  python bench/truth_t.py --bf16 work/X/bf16.gguf --imatrix work/X/imatrix.gguf \
      --text wiki.test.raw --out truth-X-q2_k.csv
"""
import argparse
import csv
import re
from pathlib import Path

from llamap.hf_gguf import GGUFFile
from llamap.llamacpp import LlamaCpp

KL = r"Mean\s+KLD:\s+([-\d.]+)\s+±\s+([\d.]+)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bf16", required=True)
    p.add_argument("--imatrix", required=True)
    p.add_argument("--text", required=True, help="evaluation text, disjoint from the imatrix calibration")
    p.add_argument("--fmt", default="q2_k")
    p.add_argument("--base-type", default="Q8_0")
    p.add_argument("--chunks", type=int, default=80)
    p.add_argument("--out", required=True)
    p.add_argument("--work", help="where to keep the base model and logits (default: next to --out)")
    p.add_argument("--llama-cpp")
    args = p.parse_args()

    llama = LlamaCpp(args.llama_cpp)
    work = Path(args.work or Path(args.out).parent)
    work.mkdir(parents=True, exist_ok=True)
    common = ["-c", 512, "--chunks", args.chunks, "-ngl", 99]
    logits = work / f"bf16-{Path(args.text).stem}-c512-{args.chunks}.kld"
    if not logits.exists():
        llama.tool("llama-perplexity", "-m", args.bf16, "-f", args.text, *common, "--kl-divergence-base", logits)

    def kl(model):
        out = llama.tool("llama-perplexity", "-m", model, *common, "--kl-divergence-base", logits, "--kl-divergence")
        return [float(v) for v in re.search(KL, out).groups()]

    base = work / f"base-{args.base_type.lower()}.gguf"
    if not base.exists():
        llama.quantize(args.bf16, base, args.base_type, args.imatrix)
    kl_base, _ = kl(base)
    print(f"base {args.base_type}: KL {kl_base:.6f}", flush=True)

    done = set()
    if Path(args.out).exists():
        with open(args.out) as f:
            done = {r["name"] for r in csv.DictReader(f)}
    bf16 = GGUFFile(args.bf16)
    names = [n for n, t in bf16.tensors.items()
             if n.startswith("blk.") and len(t.shape) >= 2 and "ffn_gate_inp" not in n and n not in done]
    variant = work / "variant.gguf"
    for name in names:
        llama.tool("llama-quantize", "--imatrix", args.imatrix, "--tensor-type", f"^{re.escape(name)}$={args.fmt}",
                   args.bf16, variant, args.base_type)
        t = GGUFFile(variant).type_of(name)
        if t != args.fmt:
            raise RuntimeError(f"{name}: got {t}, expected {args.fmt}")
        kl_v, kl_err = kl(variant)
        variant.unlink()
        layer, role = re.match(r"blk\.(\d+)\.(.+)\.weight", name).groups()
        row = {"name": name, "layer": int(layer), "role": role, "fmt": args.fmt,
               "kl": kl_v, "kl_err": kl_err, "dkl": kl_v - kl_base}
        new = not Path(args.out).exists()
        with open(args.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"{name:28s} ΔKL {row['dkl']:+.6f}", flush=True)


if __name__ == "__main__":
    main()
