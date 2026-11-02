"""Model matrices quantized in memory, straight from the BF16 GGUF.

Each matrix is quantized on demand with ggml's own functions and the same
imatrix as llama-quantize: same bytes, and no quantized model written to disk
(an 8B would otherwise need ~50 GB for the 11 candidate formats).
"""
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader

from llamap.ggml import Ggml
from llamap.hf_gguf import GGUFFile


def load_imatrix(path) -> dict[str, np.ndarray]:
    """The imatrix as llama-quantize reads it: {tensor: E[x²] per input column}.

    Two formats:
      - GGUF (recent llama-imatrix): "<name>.in_sum2" and "<name>.counts" tensors;
        E[x²] = in_sum2 / counts, per expert; an expert never seen (count 0) gets 1;
      - legacy binary (.dat): each stored sum is (raw sum / n_tokens) × ncall,
        so it is divided by ncall.
    """
    path = Path(path).expanduser()
    with open(path, "rb") as f:
        is_gguf = f.read(4) == b"GGUF"
    return _load_gguf_imatrix(path) if is_gguf else _load_legacy_imatrix(path)


def _load_gguf_imatrix(path):
    tensors = {t.name: t for t in GGUFReader(path).tensors}
    out = {}
    for key, t in tensors.items():
        if not key.endswith(".in_sum2"):
            continue
        name = key.removesuffix(".in_sum2")
        sums = np.array(t.data, dtype=np.float32).ravel()
        counts = np.array(tensors[name + ".counts"].data, dtype=np.float32).ravel()
        per_expert = sums.reshape(len(counts), -1)
        e = np.ones_like(per_expert)
        seen = counts > 0
        e[seen] = per_expert[seen] / counts[seen, None]
        out[name] = e.ravel()
    return out


def _load_legacy_imatrix(path):
    data = path.read_bytes()
    pos = 0

    def read(dtype, n=1):
        nonlocal pos
        a = np.frombuffer(data, dtype=dtype, count=n, offset=pos)
        pos += a.nbytes
        return a

    out = {}
    for _ in range(int(read(np.int32)[0])):
        name = read(np.uint8, int(read(np.int32)[0])).tobytes().decode()
        ncall = int(read(np.int32)[0])
        sums = read(np.float32, int(read(np.int32)[0])).astype(np.float32)
        out[name] = sums / ncall if ncall > 0 else sums
    return out


class QuantizedFormat:
    """One format applied in memory to every block matrix of the BF16 model, like
    llama-quantize --pure. Same interface as GGUFFile for predict.py."""

    def __init__(self, fmt: str, bf16: GGUFFile, imatrix: dict[str, np.ndarray], ggml: Ggml):
        self.fmt = fmt
        self.qtype = GGMLQuantizationType[fmt.upper()]
        self.bf16 = bf16
        self.imatrix = imatrix
        self.ggml = ggml

    def quantized_matrices(self):
        """The tensors llama-quantize quantizes in a dense model: block matrices
        (≥ 2 dims), except the MoE router (ffn_gate_inp, kept in F32)."""
        return [n for n, t in self.bf16.tensors.items()
                if n.startswith("blk.") and len(t.shape) >= 2 and "ffn_gate_inp" not in n]

    def n_bytes(self, name):
        t = self.bf16.tensors[name]
        n_per_row = int(t.shape[0])
        return int(t.n_elements) // n_per_row * self.ggml.row_size(self.qtype, n_per_row)

    def quantize(self, name) -> np.ndarray:
        t = self.bf16.tensors[name]
        n_per_row = int(t.shape[0])
        w = self.ggml(t).reshape(-1, n_per_row)
        imat = self.imatrix.get(name)
        if imat is not None and imat.size != n_per_row:
            raise ValueError(f"{name}: imatrix has {imat.size} values, expected {n_per_row} (MoE not supported)")
        return self.ggml.quantize(w, self.qtype, imat)

    def weight(self, name, shape):
        """The matrix quantized then dequantized, as float32, in GGUF row order."""
        q = self.quantize(name)
        return self.ggml.to_float(q, self.qtype, self.bf16.tensors[name].n_elements).reshape(shape)
