# llamap

Per-tensor precision maps for llama.cpp / GGUF quantization.

Given a Hugging Face model and a size budget, llamap measures how much each
weight matrix hurts the model output (KL divergence against BF16) when it alone
is quantized, in each format. It then picks one format per matrix to minimise
the total under the budget, and builds the GGUF with `llama-quantize
--tensor-type-file`.

## Usage

```
pip install -e .
export LLAMA_CPP_DIR=~/llama.cpp          # a built llama.cpp checkout

llamap map ~/models/Llama-3.1-8B-Instruct --like Q4_K_M --calib calib.txt
llamap map ~/models/Llama-3.1-8B-Instruct --like other-Q5.gguf --calib calib.txt --imatrix other.imatrix
```

`--method fisher` (default) estimates each cost as ½ E[(g·e)²] with 128 gradient
samples: linear in depth, ~2 min for the 11 formats of an 8B. `--method exact`
propagates each variant to the logits: quadratic in depth. Candidate matrices
are quantized in memory with ggml's own `ggml_quantize_chunk` and the imatrix,
byte-identical to `llama-quantize` (tested): no intermediate GGUF is written.
`--threads` sets the quantization / dequantization threads.

`--like` sets the budget: the bytes of the block matrices of a reference GGUF,
or of a llama-quantize preset. `token_embd` and `output` keep the reference's
types. Everything is cached in `work/<model>/`: changing the budget only
re-runs the solver and the build.

## Layout

| file | role |
|---|---|
| `llamap/pipeline.py` | the chain: prepare, costs, budget, solve, build |
| `llamap/predict.py` | cost of each matrix: KL at the output with only that matrix quantized |
| `llamap/hf_gguf.py` | HF ↔ GGUF names, q/k row permutation |
| `llamap/solver.py` | multiple-choice knapsack (Lagrangian relaxation + greedy fill) |
| `llamap/build.py` | `--tensor-type-file`, llama-quantize, checks |
| `llamap/quantize.py` | imatrix loading, in-memory quantization of each format |
| `llamap/ggml.py` | ctypes bindings to libggml-base (quantize, dequantize) |
| `llamap/llamacpp.py` | calls to the llama.cpp tools |
| `llamap/cli.py` | command line |
| `tests/*_t.py` | unit tests (`pytest`) |
| `bench/*_t.py` | measurements: KL against BF16, Fisher vs exact costs, additivity |

Convention: every test or measurement file ends in `_t`.
