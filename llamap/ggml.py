"""Quantization and dequantization with ggml's C library (llama.cpp's libggml-base).

Calls the same functions as llama-quantize, without going through files:
  - ggml_quantize_chunk: float32 -> quantized blocks, with the imatrix row;
  - to_float (one per type, in ggml_type_traits): blocks -> float32.
The output is byte-identical to llama-quantize (tests/ggml_t.py). ctypes releases
the GIL during these calls, so threads run truly in parallel.
"""
import ctypes
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

TO_FLOAT = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
F32P = ctypes.POINTER(ctypes.c_float)


class TypeTraits(ctypes.Structure):
    # Mirrors struct ggml_type_traits (ggml/include/ggml.h); field order matters.
    _fields_ = [
        ("type_name", ctypes.c_char_p),
        ("blck_size", ctypes.c_int64),
        ("blck_size_interleave", ctypes.c_int64),
        ("type_size", ctypes.c_size_t),
        ("is_quantized", ctypes.c_bool),
        ("to_float", ctypes.c_void_p),
        ("from_float_ref", ctypes.c_void_p),
    ]


class Ggml:
    def __init__(self, lib_path, threads=1):
        """threads: each matrix is quantized in row chunks on that many threads (rows are independent)."""
        self.lib = ctypes.CDLL(str(lib_path))
        self.pool = ThreadPoolExecutor(threads) if threads > 1 else None
        self.threads = threads
        self.lib.ggml_get_type_traits.restype = ctypes.POINTER(TypeTraits)
        self.lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
        self.lib.ggml_row_size.restype = ctypes.c_size_t
        self.lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        self.lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, F32P, ctypes.c_void_p, ctypes.c_int64,
                                                 ctypes.c_int64, ctypes.c_int64, F32P]
        self.funcs = {}

    def _to_float(self, qtype):
        if qtype not in self.funcs:
            traits = self.lib.ggml_get_type_traits(int(qtype)).contents
            # Guards against a struct layout that no longer matches ggml.h.
            if traits.type_name.decode().lower() != qtype.name.lower():
                raise RuntimeError(f"unexpected struct ggml_type_traits: {traits.type_name} for {qtype.name}")
            self.funcs[qtype] = TO_FLOAT(traits.to_float) if traits.to_float else None
        return self.funcs[qtype]

    def row_size(self, qtype, n_per_row):
        return self.lib.ggml_row_size(int(qtype), n_per_row)

    def to_float(self, data, qtype, n_elements) -> np.ndarray:
        """Raw data of a ggml type -> flat float32."""
        if qtype == GGMLQuantizationType.F32:
            return np.array(data, dtype=np.float32).ravel()
        f = self._to_float(qtype)
        if f is None:
            return dequantize(data, qtype).ravel()
        src = np.ascontiguousarray(data)
        out = np.empty(int(n_elements), dtype=np.float32)
        f(src.ctypes.data, out.ctypes.data_as(F32P), out.size)
        return out

    def __call__(self, tensor) -> np.ndarray:
        """Dequantizes a gguf-py ReaderTensor, flat."""
        return self.to_float(tensor.data, tensor.tensor_type, tensor.n_elements)

    def quantize(self, w: np.ndarray, qtype, imatrix: np.ndarray | None) -> np.ndarray:
        """w [nrows, n_per_row] float32 -> quantized bytes, as llama-quantize writes them.

        imatrix: E[x²] per input column (n_per_row values), or None.
        """
        w = np.ascontiguousarray(w, dtype=np.float32)
        nrows, n_per_row = w.shape
        row_bytes = self.row_size(qtype, n_per_row)
        out = np.empty(nrows * row_bytes, dtype=np.uint8)
        imat = None if imatrix is None else np.ascontiguousarray(imatrix, dtype=np.float32)
        imat_p = None if imat is None else imat.ctypes.data_as(F32P)

        def rows(start, stop):
            src = w[start:stop].ctypes.data_as(F32P)
            self.lib.ggml_quantize_chunk(int(qtype), src, out.ctypes.data + start * row_bytes, 0, stop - start, n_per_row, imat_p)

        if self.pool is None:
            rows(0, nrows)
        else:
            step = -(-nrows // self.threads)
            list(self.pool.map(lambda a: rows(a, min(nrows, a + step)), range(0, nrows, step)))
        return out
