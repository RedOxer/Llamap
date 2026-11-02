"""Mapping between a Hugging Face model and its GGUF files.

To put a GGUF weight in place of an HF weight, two things must match:
  1. names: "model.layers.3.mlp.down_proj.weight" in HF is "blk.3.ffn_down.weight"
     in GGUF; gguf-py provides the table;
  2. row order: for Llama-type models, the converter permutes the rows of q and k
     (interleaving the two halves of each head for llama.cpp's RoPE).
The reordering of each tensor is found by comparing the BF16 GGUF weight with the
HF weight: they must be bit-identical under one of the known reorderings.
"""
import numpy as np
from gguf import MODEL_ARCH_NAMES, GGUFReader, get_tensor_name_map
from gguf.quants import dequantize

FLOAT_TYPES = ("F32", "F16", "BF16")
# GGUF architectures tested end to end (Llama 3.1, SmolLM2, Qwen3). Others may need new
# reorderings or block call conventions, and MoE models need per-expert handling.
SUPPORTED_ARCHS = ("llama", "qwen3")


class GGUFFile:
    def __init__(self, path, dequantizer=None):
        """dequantizer: (gguf-py tensor) -> flat float32, e.g. llamap.ggml.Ggml; gguf-py's by default."""
        self.path = path
        self.reader = GGUFReader(path)
        self.tensors = {t.name: t for t in self.reader.tensors}
        self.dequantizer = dequantizer

    def field(self, key):
        return self.reader.fields[key].contents()

    def type_of(self, name):
        return self.tensors[name].tensor_type.name.lower()

    def weight(self, name, shape):
        """The weight as float32, in GGUF row order."""
        t = self.tensors[name]
        if self.dequantizer is not None:
            return self.dequantizer(t).reshape(shape)
        return dequantize(t.data, t.tensor_type).reshape(shape)

    def n_bytes(self, name):
        return int(self.tensors[name].n_bytes)

    def quantized_matrices(self):
        return [n for n, t in self.tensors.items() if n.startswith("blk.") and t.tensor_type.name not in FLOAT_TYPES]


def permute(w, n_head):
    """The converter's q/k permutation (LlamaModel.permute)."""
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def unpermute(w, n_head):
    return w.reshape(n_head, w.shape[0] // n_head // 2, 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


class WeightMapper:
    """HF parameters indexed by GGUF name, and conversion of GGUF weights to HF row order."""

    def __init__(self, model, bf16: GGUFFile):
        arch_name = bf16.field("general.architecture")
        if arch_name not in SUPPORTED_ARCHS:
            raise ValueError(f"architecture '{arch_name}' is not supported yet (supported: {', '.join(SUPPORTED_ARCHS)})")
        arch = next(a for a, n in MODEL_ARCH_NAMES.items() if n == arch_name)
        self.n_blocks = bf16.field(f"{arch_name}.block_count")
        n_head = bf16.field(f"{arch_name}.attention.head_count")
        n_head_kv = bf16.field(f"{arch_name}.attention.head_count_kv")
        name_map = get_tensor_name_map(arch, self.n_blocks)
        self.params = {}
        for hf_name, p in model.named_parameters():
            gguf_name = name_map.get_name(hf_name, try_suffixes=(".weight", ".bias"))
            if gguf_name is not None:
                self.params[gguf_name] = p
        self.bf16 = bf16
        self.reorders = {
            "none": lambda w: w,
            "llama q": lambda w: unpermute(w, n_head),
            "llama k": lambda w: unpermute(w, n_head_kv),
        }
        self.chosen = {}  # GGUF name -> verified reordering

    def check(self, name):
        """Find the reordering of name by comparing the BF16 GGUF weight with the HF weight.

        Must be called before the HF weight is swapped out, and not from a prefetch
        thread: the comparison reads the current HF weight.
        """
        if name not in self.chosen:
            w_hf = self.params[name].detach().float().cpu().numpy()
            w_ref = self.bf16.weight(name, w_hf.shape)
            match = [label for label, f in self.reorders.items() if np.array_equal(f(w_ref), w_hf)]
            if not match:
                raise ValueError(f"{name}: the BF16 GGUF weight matches the HF weight under no known reordering")
            self.chosen[name] = match[0]

    def to_hf(self, name, w_gguf):
        """A GGUF weight (quantized or not) in HF row order.

        Rows are quantized independently, so the reordering found on the BF16
        weight also holds for the quantized ones.
        """
        if name not in self.chosen:
            raise RuntimeError(f"{name}: call check() before to_hf()")
        return np.ascontiguousarray(self.reorders[self.chosen[name]](w_gguf))
