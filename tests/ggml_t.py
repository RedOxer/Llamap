"""In-memory quantization against llama-quantize: same bytes.

Needs LLAMA_CPP_DIR and a work directory with bf16.gguf and imatrix.gguf
(LLAMAP_TEST_WORK, default work/smollm2-1.7b); skipped otherwise.
"""
import os
import struct
from pathlib import Path

import numpy as np
import pytest
from gguf.quants import dequantize

from llamap.hf_gguf import GGUFFile
from llamap.llamacpp import LlamaCpp
from llamap.quantize import QuantizedFormat, load_imatrix

LLAMA = os.environ.get("LLAMA_CPP_DIR")
WORK = Path(os.environ.get("LLAMAP_TEST_WORK", Path(__file__).parents[1] / "work" / "smollm2-1.7b"))
needs_model = pytest.mark.skipif(not LLAMA or not (WORK / "bf16.gguf").exists() or not (WORK / "imatrix.gguf").exists(),
                                 reason="LLAMA_CPP_DIR or test model missing")


@pytest.fixture(scope="module")
def llama():
    return LlamaCpp(LLAMA)


def quantized_format(llama, fmt):
    ggml = llama.ggml(threads=8)
    return QuantizedFormat(fmt, GGUFFile(WORK / "bf16.gguf", ggml), load_imatrix(WORK / "imatrix.gguf"), ggml)


@needs_model
@pytest.mark.parametrize("fmt", ["q2_k", "q4_k", "q6_k", "iq2_xs"])
def test_dequantize_matches_gguf_py(llama, fmt):
    q = quantized_format(llama, fmt)
    name = "blk.3.ffn_down.weight"
    t = q.bf16.tensors[name]
    data = q.quantize(name)
    ref = dequantize(data.reshape(-1, q.ggml.row_size(q.qtype, int(t.shape[0]))), q.qtype).ravel()
    assert np.allclose(q.ggml.to_float(data, q.qtype, t.n_elements), ref, rtol=1e-6, atol=1e-7)


@needs_model
@pytest.mark.parametrize("fmt", ["q2_k", "q6_k", "iq2_xs", "iq4_xs"])
def test_same_bytes_as_llama_quantize(llama, tmp_path, fmt):
    pure = tmp_path / f"{fmt}.gguf"
    # token_embd has no imatrix entry, which IQ formats require; only block matrices are compared.
    llama.quantize(WORK / "bf16.gguf", pure, fmt.upper(), WORK / "imatrix.gguf", pure=True, token_embd="q8_0")
    q = quantized_format(llama, fmt)
    ref = GGUFFile(pure)
    assert set(q.quantized_matrices()) == set(ref.quantized_matrices())
    for n in ["blk.0.attn_q.weight", "blk.5.attn_v.weight", "blk.12.ffn_down.weight"]:
        assert q.n_bytes(n) == ref.n_bytes(n)
        assert np.array_equal(q.quantize(n), np.asarray(ref.tensors[n].data).ravel().view(np.uint8)), n


def test_legacy_imatrix_divides_by_ncall(tmp_path):
    path = tmp_path / "imatrix.dat"
    name = b"blk.0.ffn_up.weight"
    values = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    path.write_bytes(struct.pack("<i", 1) + struct.pack("<i", len(name)) + name
                     + struct.pack("<ii", 2, len(values)) + values.tobytes())
    assert np.array_equal(load_imatrix(path)["blk.0.ffn_up.weight"], values / 2)
