"""Measurement: heatmaps of the format chosen for each block matrix (layer × role).

Each argument is label=path, the path being a GGUF (types read from the file) or a
--tensor-type-file written by llamap. Colour encodes bits per weight.

  python bench/plot_maps_t.py --out docs/maps.png unsloth=X-UD-Q5_K_XL.gguf llamap=work/X/map-Q5.tensor-types.txt
"""
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from llamap.hf_gguf import GGUFFile

BPW = {"q2_k": 2.625, "iq2_xs": 2.3125, "iq2_s": 2.5625, "iq3_xxs": 3.0625, "q3_k": 3.4375, "iq3_s": 3.4375,
       "iq4_xs": 4.25, "iq4_nl": 4.5, "q4_0": 4.5, "q4_1": 5.0, "q4_k": 4.5, "q5_0": 5.5, "q5_1": 6.0, "q5_k": 5.5,
       "q6_k": 6.5625, "q8_0": 8.5, "iq1_s": 1.5625, "iq1_m": 1.75, "iq2_xxs": 2.0625, "bf16": 16.0, "f16": 16.0}
ROLES = ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def read_map(path):
    if str(path).endswith(".gguf"):
        f = GGUFFile(path)
        return {n: f.type_of(n) for n in f.quantized_matrices()}
    pairs = (line.strip().split("=") for line in open(path) if line.strip())
    return {k.replace("\\", "")[1:-1]: v for k, v in pairs}


def grid(choice):
    layers = 1 + max(int(re.match(r"blk\.(\d+)\.", n)[1]) for n in choice)
    g = np.full((len(ROLES), layers), np.nan)
    for n, t in choice.items():
        m = re.match(r"blk\.(\d+)\.(\w+)\.weight", n)
        if m and m[2] in ROLES:
            g[ROLES.index(m[2]), int(m[1])] = BPW[t]
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="")
    p.add_argument("maps", nargs="+", help="label=path")
    args = p.parse_args()

    maps = [(s.split("=", 1)[0], grid(read_map(s.split("=", 1)[1]))) for s in args.maps]
    cmap = LinearSegmentedColormap.from_list("bpw", RAMP)
    ink, muted = "#2b2b29", "#6b6b66"
    plt.rcParams.update({"font.size": 10, "text.color": ink, "axes.labelcolor": muted,
                         "xtick.color": muted, "ytick.color": muted})
    fig, axes = plt.subplots(len(maps), 1, figsize=(11, 1.6 * len(maps) + 0.9), facecolor="#fcfcfb",
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (label, g) in zip(axes, maps):
        im = ax.imshow(g, cmap=cmap, vmin=2.3, vmax=8.5, aspect="auto", interpolation="nearest")
        # 2 px surface gaps between cells
        ax.set_xticks(np.arange(-0.5, g.shape[1]), minor=True)
        ax.set_yticks(np.arange(-0.5, g.shape[0]), minor=True)
        ax.grid(which="minor", color="#fcfcfb", linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        ax.set_yticks(range(len(ROLES)), ROLES)
        ax.set_title(label, loc="left", fontsize=11, color=ink)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axes[-1].set_xticks(range(0, maps[0][1].shape[1], 2))
    axes[-1].set_xlabel("layer")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.01, aspect=30)
    ticks = {"IQ2_XS": 2.3125, "Q3_K": 3.4375, "IQ4_XS": 4.25, "Q5_K": 5.5, "Q6_K": 6.5625, "Q8_0": 8.5}
    cb.set_ticks(list(ticks.values()), labels=[f"{k} ({v:.2f})" for k, v in ticks.items()])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=9)
    cb.set_label("bits per weight")
    if args.title:
        fig.suptitle(args.title, x=0.01, ha="left", fontsize=13, color=ink)
    fig.savefig(args.out, dpi=150, facecolor=fig.get_facecolor())
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
