"""Measurement: GGUF files against the BF16 logits, on several texts.

For each model and text: mean KL (± error), median, 99th and 99.9th percentiles,
maximum, share of tokens with the same top-1 as BF16, and |PPL - PPL_BF16|. The
BF16 logits (.kld) are computed once and kept; they include the tokens, so every
model sees exactly the same ones.

Texts in --special-texts (chat-template text) get their BF16 logits from
bench/perplexity (llamap-perplexity), which parses the template's special tokens;
stock llama-perplexity tokenizes them as plain text. Only the BF16 run needs it:
the quantized models are then scored on the tokens stored in the .kld file.

  python bench/eval_kl_t.py --bf16 work/X/bf16.gguf --texts wiki.test.raw,code.txt \
      --out results.csv unsloth=~/models/X-UD-Q4_K_XL.gguf llamap=work/X/llamap-X-UD-Q4_K_XL.gguf
"""
import argparse
import csv
import re
from pathlib import Path

from llamap.llamacpp import LlamaCpp

FIELDS = {
    "kl": r"Mean\s+KLD:\s+([-\d.]+)",
    "kl_err": r"Mean\s+KLD:\s+[-\d.]+\s+±\s+([\d.]+)",
    "kl_median": r"Median\s+KLD:\s+([-\d.]+)",
    "kl_p99": r"99\.0%\s+KLD:\s+([-\d.]+)",
    "kl_p999": r"99\.9%\s+KLD:\s+([-\d.]+)",
    "kl_max": r"Maximum\s+KLD:\s+([-\d.]+)",
    "same_top": r"Same top p:\s+([\d.]+)",
    "ppl": r"Mean\s+PPL\(Q\)\s*:\s*([\d.]+)",
    "ppl_bf16": r"Mean\s+PPL\(base\)\s*:\s*([\d.]+)",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bf16", required=True)
    p.add_argument("--texts", default="", help="comma-separated text files")
    p.add_argument("--special-texts", default="", help="comma-separated text files with special tokens")
    p.add_argument("--special-perplexity", default=Path(__file__).parent / "perplexity/build/llamap-perplexity")
    p.add_argument("--kld-dir", help="where to keep the BF16 logits (default: next to the BF16)")
    p.add_argument("--chunks", type=int, default=80)
    p.add_argument("--out", required=True)
    p.add_argument("--llama-cpp")
    p.add_argument("models", nargs="+", help="label=path.gguf")
    args = p.parse_args()

    llama = LlamaCpp(args.llama_cpp)
    kld_dir = Path(args.kld_dir or Path(args.bf16).parent / "kld")
    kld_dir.mkdir(parents=True, exist_ok=True)
    common = ["-c", 512, "--chunks", args.chunks, "-ngl", 99]
    rows = []
    special = [t for t in args.special_texts.split(",") if t]
    for text in [t for t in args.texts.split(",") if t] + special:
        tool = args.special_perplexity if text in special else llama.bin / "llama-perplexity"
        text = Path(text).expanduser()
        base = kld_dir / f"{text.stem}-c512-{args.chunks}.kld"
        if not base.exists():
            llama.run([tool, "-m", args.bf16, "-f", text, *common, "--kl-divergence-base", base])
        for spec in args.models:
            label, path = spec.split("=", 1)
            out = llama.tool("llama-perplexity", "-m", Path(path).expanduser(), *common,
                             "--kl-divergence-base", base, "--kl-divergence")
            r = {"text": text.stem, "label": label, **{k: float(re.search(rx, out).group(1)) for k, rx in FIELDS.items()}}
            r["ppl_gap"] = abs(r["ppl"] - r["ppl_bf16"])
            rows.append(r)
            print(f"{text.stem:10s} {label:20s} KL {r['kl']:.5f} ± {r['kl_err']:.5f}  p99.9 {r['kl_p999']:.4f}  "
                  f"max {r['kl_max']:.3f}  same top {r['same_top']:.2f} %  |ΔPPL| {r['ppl_gap']:.3f}", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
