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
samples: linear in depth, ~15 min for the 11 formats of an 8B (RTX 5090, 16
CPU threads), most of it spent quantizing the candidates. `--method exact`
propagates each variant to the logits: quadratic in depth. Candidate matrices
are quantized in memory with ggml's own `ggml_quantize_chunk` and the imatrix,
byte-identical to `llama-quantize` (tested): no intermediate GGUF is written.
`--threads` sets the quantization / dequantization threads.

`--like` sets the budget: the bytes of the block matrices of a reference GGUF,
or of a llama-quantize preset. `token_embd` and `output` keep the reference's
types. Everything is cached in `work/<model>/`: changing the budget only
re-runs the solver and the build.

## Results

Three maps at the same file sizes, the sizes of Unsloth's Dynamic 2.0
`UD-Qn_K_XL` files:
- [Unsloth Dynamic 2.0](https://huggingface.co/unsloth), the published files;
- the `--target-size` mode of [llama.cpp PR #15550](https://github.com/ggml-org/llama.cpp/pull/15550)
  (EAddario), which estimates each tensor's error from its weights;
- llamap (Fisher, 128 samples).

EAddario and llamap use the same imatrix, and the sizes match within 0.2 MB.
`token_embd` and `output` are identical in all three, so only the block matrices differ.

KL divergence against BF16 (`llama-perplexity --kl-divergence`, 80 × 512 tokens
per text), averaged over 6 texts for Llama and 7 for Qwen3. The texts are
wikitext test and valid, a book, code, chat, chat with the model's template,
and Qwen3 thinking traces. In brackets: the number of texts on which llamap is
better.

| Model | Budget | Unsloth | EAddario | llamap | vs Unsloth | vs EAddario |
|---|---|---|---|---|---|---|
| Llama 3.1 8B Instruct | Q2 (3232 MiB) | 0.2633 | 0.1804 | **0.1666** | −37 % (6/6) | −8 % (6/6) |
| | Q3 (4005 MiB) | 0.0886 | 0.0681 | **0.0653** | −26 % (6/6) | −4 % (5/6) |
| | Q4 (4763 MiB) | 0.0327 | 0.0283 | **0.0272** | −17 % (6/6) | −4 % (4/6) |
| | Q5 (5472 MiB) | 0.0110 | 0.0111 | **0.0096** | −13 % (6/6) | −14 % (6/6) |
| Qwen3-8B | Q2 (3340 MiB) | 0.1848 | 0.1228 | **0.1199** | −35 % (7/7) | −2 % (4/7) |
| | Q3 (4107 MiB) | 0.0639 | 0.0433 | **0.0392** | −39 % (7/7) | −9 % (6/7) |
| | Q4 (4898 MiB) | 0.0192 | 0.0170 | **0.0153** | −20 % (7/7) | −10 % (7/7) |
| | Q5 (5605 MiB) | 0.0075 | 0.0078 | **0.0058** | −23 % (7/7) | −25 % (7/7) |

- The gap with EAddario is largest at high precision (Q5). There, an error
  estimated from the weights ranks tensors poorly: on Llama, the last layers cost
  ~3× more in output KL for the same relative error.
- Inference speed is unchanged. llama-bench on CUDA gives the same pp512 within
  1 %, and tg128 at most 4 % lower (Q2, against Unsloth).
- Build time on an 8B: llamap takes ~15 min for the first budget (cost
  measurement), then ~1 min per extra budget. EAddario takes ~3 min, then
  ~1 min per extra budget.
- Limits: dense models only (no MoE yet); needs PyTorch, the HF weights and a
  GPU; two models tested.

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
