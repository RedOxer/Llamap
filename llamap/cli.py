"""Command line.

  llamap map ~/models/Llama-3.1-8B-Instruct --like Q4_K_M --calib calib.txt
  llamap map ~/models/Llama-3.1-8B-Instruct --like other-Q5.gguf --calib calib.txt --imatrix other.imatrix
  llamap map ~/models/SmolLM2-1.7B --like Q4_K_M --calib calib.txt --method exact
"""
import argparse
import os
import time
from collections import Counter

from llamap.llamacpp import LlamaCpp
from llamap.pipeline import FORMATS, Pipeline


def main(argv=None):
    p = argparse.ArgumentParser(prog="llamap", description="Per-tensor precision maps for llama.cpp / GGUF")
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("map", help="compute the map under a budget and build the GGUF")
    m.add_argument("hf_dir", help="Hugging Face model directory")
    m.add_argument("--like", required=True, help="budget: a reference GGUF, or a llama-quantize preset (Q4_K_M...)")
    m.add_argument("--calib", required=True, help="calibration text (imatrix and cost prediction)")
    m.add_argument("--imatrix", help="existing imatrix (otherwise computed on --calib)")
    m.add_argument("--formats", default=",".join(FORMATS))
    m.add_argument("--work", help="work directory (default: work/<model name>)")
    m.add_argument("--out", help="output GGUF (default: in the work directory)")
    m.add_argument("--llama-cpp", help="llama.cpp directory (default: LLAMA_CPP_DIR)")
    m.add_argument("--n-tokens", type=int, default=512, help="calibration tokens for cost prediction")
    m.add_argument("--threads", type=int, default=os.cpu_count(), help="quantization threads")
    m.add_argument("--method", choices=["fisher", "exact"], default="fisher",
                   help="fisher: linear in depth (default); exact: KL propagated to the logits, slower")
    m.add_argument("--samples", type=int, default=128, help="Fisher samples (32 is too noisy)")
    args = p.parse_args(argv)

    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f} s] {msg}", flush=True)

    work = args.work or f"work/{args.hf_dir.rstrip('/').split('/')[-1]}"
    pipe = Pipeline(args.hf_dir, work, args.calib, LlamaCpp(args.llama_cpp), args.imatrix, args.n_tokens, args.threads, log)
    choice, r = pipe.run(args.like, args.formats.split(","), args.out, args.method, args.samples)

    mib = 2 ** 20
    log(f"map: {r['map']}")
    log(f"GGUF: {r['gguf']} ({r['file_bytes'] / mib:.1f} MiB, reference {r['ref_file_bytes'] / mib:.1f} MiB)")
    log(f"formats: {dict(Counter(choice.values()).most_common())}")
    ref_kl = f" (reference {r['ref_predicted_kl']:.5f})" if r["ref_predicted_kl"] is not None else ""
    log(f"predicted KL: {r['predicted_kl']:.5f}{ref_kl}")


if __name__ == "__main__":
    main()
