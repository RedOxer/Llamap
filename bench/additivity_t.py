"""Measurement: additivity of the costs, the solver's assumption.

For each map, on the calibration text:
  predicted = Σ_i kl(i, t_i), the sum of the individual costs
  real      = output KL with every matrix of the map quantized at once
If real / predicted varies little between maps, the solver ranks maps correctly.
Random maps serve as controls.

  python bench/additivity_t.py --hf-dir ~/models/X --work work/X --calib calib.txt \
      work/X/map-Q4_K_M.tensor-types.txt
"""
import argparse
import random
import re
from pathlib import Path

from llamap.hf_gguf import GGUFFile
from llamap.llamacpp import LlamaCpp
from llamap.pipeline import load_costs
from llamap.predict import Replayer
from llamap.quantize import QuantizedFormat, load_imatrix


def read_map(path):
    pairs = (line.strip().split("=") for line in open(path) if line.strip())
    return {re.sub(r"\\", "", k)[1:-1]: v for k, v in pairs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dir", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--calib", required=True)
    p.add_argument("--imatrix", help="default: the work directory's")
    p.add_argument("--random", type=int, default=2)
    p.add_argument("--costs", help="costs to test (default: work/costs-exact.csv)")
    p.add_argument("--llama-cpp")
    p.add_argument("maps", nargs="+")
    args = p.parse_args()

    work = Path(args.work)
    imatrix = args.imatrix or work / "imatrix.gguf"
    costs = {(c.name, c.fmt): c.kl for c in load_costs(args.costs or work / "costs-exact.csv")}
    maps = {Path(m).name.removesuffix(".tensor-types.txt"): read_map(m) for m in args.maps}
    used = sorted({t for m in maps.values() for t in m.values()})
    names = sorted(next(iter(maps.values())))
    rng = random.Random(0)
    for k in range(args.random):
        maps[f"random-{k}"] = {n: rng.choice(used) for n in names}

    llama = LlamaCpp(args.llama_cpp)
    ggml = llama.ggml()
    bf16 = GGUFFile(work / "bf16.gguf", ggml)
    imat = load_imatrix(imatrix)
    formats = {t: QuantizedFormat(t, bf16, imat, ggml) for t in used}
    r = Replayer(args.hf_dir, work / "bf16.gguf", Path(args.calib).read_text(), dequantizer=ggml)
    print(f"{'map':36s} {'predicted':>10s} {'real':>10s} {'real/pred':>12s}")
    for label, choice in maps.items():
        predicted = sum(costs[n, t] for n, t in choice.items())
        real = r.kl_of_map(choice, formats)
        print(f"{label:36s} {predicted:10.5f} {real:10.5f} {real / predicted:12.2f}", flush=True)


if __name__ == "__main__":
    main()
