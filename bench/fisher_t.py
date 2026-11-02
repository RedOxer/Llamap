"""Measurement: Fisher costs against exact costs.

Both files come from `llamap map --method fisher` and `--method exact` on the same
work directory. Prints the Spearman correlation per format and within each role,
and the Fisher / exact ratio.

  python bench/fisher_t.py work/X/costs-fisher.csv work/X/costs-exact.csv
"""
import argparse

import pandas as pd


def spearman(x, y):
    return x.rank().corr(y.rank())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("fisher")
    p.add_argument("exact")
    args = p.parse_args()

    d = pd.read_csv(args.fisher).merge(pd.read_csv(args.exact), on=["name", "fmt"], suffixes=("_fisher", "_exact"))
    d["role"] = d.name.str.extract(r"blk\.\d+\.(.+)\.weight")[0]
    print(f"{len(d)} variants: Spearman {spearman(d.kl_fisher, d.kl_exact):.3f}, "
          f"median Fisher / exact {(d.kl_fisher / d.kl_exact).median():.2f}")
    for fmt, g in d.groupby("fmt", sort=False):
        roles = [spearman(r.kl_fisher, r.kl_exact) for _, r in g.groupby("role")]
        print(f"  {fmt:8s} all {spearman(g.kl_fisher, g.kl_exact):.3f}   within role {min(roles):.2f} … {max(roles):.2f}")


if __name__ == "__main__":
    main()
