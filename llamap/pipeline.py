"""The full chain, from an HF model to a GGUF built from our map.

  1. prepare: BF16 GGUF and imatrix (unless one is given);
  2. costs:   output KL of each matrix in each format, quantized in memory;
  3. budget:  bytes of the block matrices of a reference GGUF (file or preset);
  4. solve:   one format per matrix under that budget;
  5. build:   the final GGUF, checked.
Everything lives in a work directory and each step reuses what is already there:
changing the budget only re-runs steps 3 to 5.
"""
import csv
import os
from dataclasses import astuple, fields
from pathlib import Path

from llamap.build import build
from llamap.hf_gguf import GGUFFile
from llamap.llamacpp import LlamaCpp
from llamap.predict import Cost, Replayer
from llamap.quantize import QuantizedFormat, load_imatrix
from llamap.solver import solve, total

FORMATS = ["q2_k", "iq2_xs", "iq2_s", "q3_k", "iq3_xxs", "iq3_s", "iq4_xs", "q4_k", "q5_k", "q6_k", "q8_0"]


def load_costs(path) -> list[Cost]:
    if not Path(path).exists():
        return []
    with open(path) as f:
        return [Cost(r["name"], r["fmt"], float(r["kl"]), float(r["block_err"]), int(r["n_bytes"]), float(r.get("kl_se") or 0))
                for r in csv.DictReader(f)]


def append_costs(path, costs: list[Cost]):
    new = not Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow([fl.name for fl in fields(Cost)])
        w.writerows(astuple(c) for c in costs)


def tables(costs: list[Cost]):
    """cost and size indexed [matrix][format], for the solver."""
    cost, size = {}, {}
    for c in costs:
        cost.setdefault(c.name, {})[c.fmt] = c.kl
        size.setdefault(c.name, {})[c.fmt] = c.n_bytes
    return cost, size


class Pipeline:
    def __init__(self, hf_dir, work, calib, llama: LlamaCpp, imatrix=None, n_tokens=512, threads=1, log=print):
        self.hf_dir = Path(hf_dir).expanduser()
        self.work = Path(work).expanduser()
        self.work.mkdir(parents=True, exist_ok=True)
        self.calib = Path(calib).expanduser()
        self.llama = llama
        self.bf16 = self.work / "bf16.gguf"
        self.imatrix = Path(imatrix).expanduser() if imatrix else self.work / "imatrix.gguf"
        self.n_tokens = n_tokens
        self.threads = threads
        self.ggml = llama.ggml(threads)
        self.log = log

    def prepare(self):
        if not self.bf16.exists():
            self.log("converting to BF16 GGUF")
            self.llama.convert(self.hf_dir, self.bf16)
        if not self.imatrix.exists():
            self.log("computing the imatrix")
            self.llama.imatrix(self.bf16, self.calib, self.imatrix)

    def formats(self, formats) -> dict[str, QuantizedFormat]:
        bf16 = GGUFFile(self.bf16, self.ggml)
        imatrix = load_imatrix(self.imatrix)
        return {f: QuantizedFormat(f, bf16, imatrix, self.ggml) for f in formats}

    def costs(self, formats=FORMATS, method="fisher", samples=128) -> list[Cost]:
        """Cost of each matrix in each format, cached in costs-<method>.csv."""
        path = self.work / f"costs-{method}.csv"
        done = {c.fmt for c in load_costs(path)}
        todo = [f for f in formats if f not in done]
        if todo:
            replayer = Replayer(self.hf_dir, self.bf16, self.calib.read_text(), self.n_tokens,
                                threads=self.threads, dequantizer=self.ggml)
            if method == "exact":
                self.log(f"exact KL, {len(todo)} formats")
                new = replayer.costs(self.formats(todo), progress=self.log)
            elif method == "fisher":
                self.log(f"Fisher, {samples} samples, {len(todo)} formats")
                new = replayer.fisher_costs(self.formats(todo), samples, progress=self.log)
            else:
                raise ValueError(f"unknown method: {method}")
            append_costs(path, new)
        return [c for c in load_costs(path) if c.fmt in formats]

    def reference(self, like) -> GGUFFile:
        """An existing GGUF, or a llama-quantize preset (Q4_K_M...) built here."""
        if Path(like).expanduser().exists():
            return GGUFFile(Path(like).expanduser())
        path = self.work / f"ref-{like}.gguf"
        if not path.exists():
            self.log(f"building preset {like} for its budget")
            self.llama.quantize(self.bf16, path, like, self.imatrix)
        return GGUFFile(path)

    def run(self, like, formats=FORMATS, out=None, method="fisher", samples=128):
        self.prepare()
        cost, size = tables(self.costs(formats, method, samples))
        ref = self.reference(like)
        budget = sum(int(ref.tensors[n].n_bytes) for n in cost)
        choice = solve(cost, size, budget)

        label = (Path(like).stem if Path(like).expanduser().exists() else like) + f"-{method}"
        map_path = self.work / f"map-{label}.tensor-types.txt"
        out = Path(out).expanduser() if out else self.work / f"llamap-{label}.gguf"
        build(self.llama, self.bf16, self.imatrix, choice, ref, map_path, out)

        ref_choice = {n: ref.type_of(n) for n in cost}
        report = {
            "map": map_path,
            "gguf": out,
            "budget_bytes": budget,
            "map_bytes": total(choice, size),
            "file_bytes": os.path.getsize(out),
            "ref_file_bytes": os.path.getsize(ref.path),
            "predicted_kl": total(choice, cost),
            # Only defined when every format of the reference has been measured.
            "ref_predicted_kl": total(ref_choice, cost) if all(t in formats for t in ref_choice.values()) else None,
        }
        return choice, report
