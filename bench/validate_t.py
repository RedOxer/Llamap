"""Measurement: predicted costs against the degrade-one ground truth (bench/truth_t.py).

For each predictor: Spearman correlation with the measured ΔKL over all block
matrices, the range of the correlation within each role, and the depth bias
(correlation between the layer index and log(ΔKL / prediction)).

Predictors:
  --costs label=costs.csv   llamap costs (`llamap costs`): the kl column, and
                            block_err (error at the block output) for the first file;
  --eaddario-log FILE       EAddario's own per-tensor error estimates (PR #15550),
                            from the log of (all types excluded but --fmt):
                            llama-quantize --imatrix X.imatrix --target-bpw 3 \\
                                --target-exclude <every other type> X-bf16.gguf out.gguf
  --bf16 + --imatrix        Σ imatrix_j (Δw_ij)², an imatrix-weighted weight error
                            (our own simple version, not EAddario's metric).

  python bench/validate_t.py --truth truth-X-q2_k.csv --costs exact=work/X/costs-exact.csv \
      --costs fisher=work/X/costs-fisher.csv --eaddario-log eaddario.log --bf16 work/X/bf16.gguf --imatrix work/X/imatrix.gguf
"""
import argparse
import re

import numpy as np
import pandas as pd


def spearman(x, y):
    return pd.Series(np.asarray(x, dtype=float)).rank().corr(pd.Series(np.asarray(y, dtype=float)).rank())


def read_eaddario_log(path, fmt):
    """{tensor: error} from the log of llama-quantize --target-bpw (PR #15550) run with
    --target-exclude set to every type but fmt, so that fmt is each tensor's choice and its
    error is printed in the "tensor quantization mix". These are the same values as in a
    normal run's --state-file, which keeps only each tensor's convex-hull candidates."""
    pattern = rf"(\S+\.weight) \S*\s+{fmt},.*?error: ([\d.]+)"
    return {m[1]: float(m[2]) for m in re.finditer(pattern, open(path).read(), re.IGNORECASE)}


def imatrix_weight_error(names, bf16_path, imatrix_path, fmt):
    from llamap.hf_gguf import GGUFFile
    from llamap.llamacpp import LlamaCpp
    from llamap.quantize import QuantizedFormat, load_imatrix
    ggml = LlamaCpp().ggml(threads=16)
    q = QuantizedFormat(fmt, GGUFFile(bf16_path, ggml), load_imatrix(imatrix_path), ggml)
    out = {}
    for n in names:
        t = q.bf16.tensors[n]
        shape = (-1, int(t.shape[0]))
        dw = q.weight(n, shape).astype(np.float64) - ggml(t).reshape(shape)
        out[n] = float((dw ** 2 * q.imatrix[n]).sum())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--truth", required=True)
    p.add_argument("--costs", action="append", default=[], help="label=costs.csv")
    p.add_argument("--eaddario-log")
    p.add_argument("--bf16")
    p.add_argument("--imatrix")
    args = p.parse_args()

    d = pd.read_csv(args.truth).rename(columns={"qtype": "fmt"})  # older truth files
    fmt = d.fmt.iloc[0]
    preds = {}
    if args.bf16 and args.imatrix:
        preds["imatrix-weighted weight error"] = imatrix_weight_error(d.name, args.bf16, args.imatrix, fmt)
    if args.eaddario_log:
        preds["EAddario --target-bpw error"] = read_eaddario_log(args.eaddario_log, fmt)
    for i, spec in enumerate(args.costs):
        label, path = spec.split("=", 1)
        c = pd.read_csv(path)
        c = c[c.fmt == fmt].set_index("name")
        if i == 0:
            preds["error at the block output"] = c.block_err.to_dict()
        preds[f"output KL, {label}"] = c.kl.to_dict()

    print(f"{len(d)} tensors, {fmt}; Spearman with the measured ΔKL")
    print(f"  {'predictor':32s} {'n':>4s} {'all':>6s} {'within role':>14s} {'depth bias':>11s}")
    for label, pred in preds.items():
        m = d[d.name.isin(pred.keys())].copy()
        m["pred"] = m.name.map(pred)
        roles = [spearman(g.pred, g.dkl) for _, g in m.groupby("role") if len(g) > 2]
        pos = m[(m.dkl > 0) & (m.pred > 0)]
        bias = np.corrcoef(pos.layer, np.log(pos.dkl / pos.pred))[0, 1]
        print(f"  {label:32s} {len(m):4d} {spearman(m.pred, m.dkl):6.3f} "
              f"{min(roles):+6.2f}…{max(roles):+5.2f} {bias:+11.2f}")


if __name__ == "__main__":
    main()
