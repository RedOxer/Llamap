"""Measurement: stability of the costs and maps across calibration texts.

Each work directory comes from `llamap map` with the same model, imatrix and
budgets, and a different --calib. Prints the Spearman correlation between the
costs of each pair of calibrations (all tensor × format pairs, then the mean of
the per-format correlations), and for each budget the number of tensors whose
format differs from the first calibration's map.

  python bench/calib_t.py wiki=work/X wiki2=work/calib-wiki2 chat=work/calib-chat code=work/calib-code
"""
import argparse
import itertools
import re
from pathlib import Path

import pandas as pd


def spearman(x, y):
    return x.rank().corr(y.rank())


def read_map(path):
    return {re.sub(r"\\(.)", r"\1", k.strip("^$")): v
            for k, v in (line.strip().rsplit("=", 1) for line in open(path) if line.strip())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("works", nargs="+", help="label=work directory")
    p.add_argument("--method", default="fisher")
    args = p.parse_args()
    works = dict(w.split("=", 1) for w in args.works)

    costs = {k: pd.read_csv(Path(w) / f"costs-{args.method}.csv").set_index(["name", "fmt"]).kl
             for k, w in works.items()}
    print("Spearman between the costs")
    print(f"  {'pair':14s} {'all':>6s} {'per format (mean, min)':>24s}")
    for a, b in itertools.combinations(works, 2):
        d = pd.concat([costs[a].rename("a"), costs[b].rename("b")], axis=1, join="inner").reset_index()
        per_fmt = [spearman(g.a, g.b) for _, g in d.groupby("fmt")]
        print(f"  {a + '/' + b:14s} {spearman(d.a, d.b):6.3f} "
              f"{sum(per_fmt) / len(per_fmt):16.3f} {min(per_fmt):6.3f}")

    first = next(iter(works))
    maps = {k: {re.search(r"-(Q\d)-", f.name)[1]: read_map(f)
                for f in Path(w).glob(f"map-*-{args.method}.tensor-types.txt")}
            for k, w in works.items()}
    print(f"\nTensors whose format differs from the {first} map")
    budgets = sorted(maps[first])
    print(f"  {'calib':8s} " + " ".join(f"{b:>9s}" for b in budgets))
    for k in works:
        cells = []
        for b in budgets:
            ref, m = maps[first][b], maps[k].get(b)
            cells.append(f"{sum(m[n] != ref[n] for n in ref):4d}/{len(ref)}" if m else "-")
        print(f"  {k:8s} " + " ".join(f"{c:>9s}" for c in cells))


if __name__ == "__main__":
    main()
