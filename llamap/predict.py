"""Cost of each matrix in each format: the KL divergence at the model output when
only that matrix is quantized.

Exact method, for a matrix W of block i and a format t:
  1. start from the input of block i, captured once on the calibration text;
  2. swap W for its quantized version, replay block i, restore W;
  3. propagate through blocks i+1 … end, the final norm and lm_head;
  4. KL(reference logits ‖ these logits), averaged over tokens.
The reference is the same path without the swap, so the two differ only by the
quantized matrix. This is quadratic in depth.

Fisher method: replay only block i and approximate the KL by ½ E_y[(g·e)²], with
e the error at the output of block i and g the gradient of log p(y) with respect
to it, y sampled from the model. One backward pass gives g for every block, so
this is linear in depth.

An error measured at the block output alone (‖e‖² / ‖h‖², kept in block_err for
comparison) assumes that the same relative error costs the same at any depth.
It does not: on Llama-type models the last layers cost ~3× more.

The model is loaded in BF16 (an 8B in float32 does not fit in 32 GB), but each
block is cast to float32 while it is replayed: BF16 rounding of the activations
(~0.4 %) would drown the small errors of Q6_K / Q8_0.
"""
import contextlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llamap.hf_gguf import GGUFFile, WeightMapper


@dataclass
class Cost:
    name: str
    fmt: str
    kl: float           # KL at the output: what the solver minimises
    block_err: float    # E_t[‖e_t‖² / ‖h_t‖²] at the block output
    n_bytes: int
    kl_se: float = 0.0  # standard error of kl from Fisher sampling; 0 for the exact method


def _to_f32(x):
    if torch.is_tensor(x):
        return x.float() if x.is_floating_point() else x
    if isinstance(x, (tuple, list)):
        return type(x)(_to_f32(v) for v in x)
    if isinstance(x, dict):
        return {k: _to_f32(v) for k, v in x.items()}
    return x


class Replayer:
    """An HF model, its calibration tokens and the captured inputs of each block."""

    def __init__(self, hf_dir, bf16_gguf, text, n_tokens=512, chunk=8, device="cuda", threads=1, dequantizer=None,
                 prefetch_gb=2.0):
        """threads: matrices quantized ahead in parallel.
        prefetch_gb: RAM cap for matrices quantized ahead (one 8B FFN matrix: 235 MB)."""
        torch.backends.cuda.matmul.allow_tf32 = False  # TF32 only has a 10-bit mantissa
        self.dtype = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=self.dtype, device_map=device).eval()
        tokenizer = AutoTokenizer.from_pretrained(hf_dir)
        # A token is rarely longer than 16 characters: no need to tokenize a whole large text.
        self.ids = tokenizer(text[:16 * n_tokens], return_tensors="pt").input_ids[:, :n_tokens].to(device)
        self.mapper = WeightMapper(self.model, GGUFFile(bf16_gguf, dequantizer))
        self.threads = threads
        self.prefetch_bytes = prefetch_gb * 2 ** 30
        self.pool = ThreadPoolExecutor(threads) if threads > 1 else None
        self.chunk = chunk

        n = self.mapper.n_blocks
        self.blocks = next(m for m in self.model.modules() if isinstance(m, torch.nn.ModuleList) and len(m) == n)
        self.norm = self.model.base_model.norm
        self.lm_head = self.model.get_output_embeddings()

        # One full pass; a hook keeps the exact arguments of each block (input,
        # positions, mask). Only the input changes afterwards.
        self.calls = {}
        handles = [b.register_forward_pre_hook(self._keep(i), with_kwargs=True) for i, b in enumerate(self.blocks)]
        with torch.no_grad():
            self.model(self.ids, use_cache=False)
        for h in handles:
            h.remove()
        # Only after the capture: lm_head may share its weight with the embeddings.
        self.norm.float()
        self.lm_head.float()

    def _keep(self, i):
        def hook(module, args, kwargs):
            self.calls[i] = _to_f32((args, kwargs))
        return hook

    @contextlib.contextmanager
    def _float32(self, i):
        self.blocks[i].float()
        try:
            yield
        finally:
            self.blocks[i].to(self.dtype)  # lossless: the original weights are BF16

    def _run_block(self, i, hidden):
        args, kwargs = self.calls[i]
        with torch.no_grad():
            out = self.blocks[i](hidden, *args[1:], **kwargs)
        return out[0] if isinstance(out, tuple) else out

    def _log_probs(self, hidden):
        with torch.no_grad():
            return torch.log_softmax(self.lm_head(self.norm(hidden)).double(), dim=-1)

    def _kl_from(self, i, hidden):
        """hidden[0] = reference, hidden[k] = variant k, all at the output of block i.
        Returns the output KL of each variant (k ≥ 1)."""
        for j in range(i + 1, len(self.blocks)):
            with self._float32(j):
                hidden = torch.cat([self._run_block(j, h) for h in hidden.split(self.chunk)])
        ref = self._log_probs(hidden[:1])
        kls = []
        for h in hidden[1:].split(self.chunk):
            lp = self._log_probs(h)
            kls += (ref.exp() * (ref - lp)).sum(-1).mean(-1).tolist()
        return kls

    def _load(self, name, f: GGUFFile):
        return self.mapper.to_hf(name, f.weight(name, tuple(self.mapper.params[name].shape)))

    def _weights(self, items):
        """The quantized weights of items [(name, format)], in order, prepared in parallel.
        Matrices prepared ahead stay under prefetch_bytes (and 2 × threads): without
        this cap an 8B runs out of RAM."""
        if self.pool is None:
            yield from (self._load(n, f) for n, f in items)
            return
        nbytes = lambda n: 4 * self.mapper.params[n].numel()
        pending, window, used = deque(items), deque(), 0
        while pending or window:
            while pending and len(window) < 2 * self.threads and (not window or used + nbytes(pending[0][0]) <= self.prefetch_bytes):
                n, f = pending.popleft()
                window.append((n, self.pool.submit(self._load, n, f)))
                used += nbytes(n)
            n, fut = window.popleft()
            used -= nbytes(n)
            yield fut.result()

    def _block_variants(self, i, formats, names):
        """Outputs of block i, float32: [reference, then one per (matrix, format)]."""
        keys = [(n, f) for n in names if n.startswith(f"blk.{i}.") for f in formats]
        with self._float32(i):
            x = self.calls[i][0][0]
            outs = [self._run_block(i, x)]
            if i + 1 < len(self.blocks):
                # The float32 replay must reproduce the next block's captured input,
                # up to the BF16 rounding of the capture pass.
                nxt = self.calls[i + 1][0][0]
                snr = 10 * torch.log10((nxt ** 2).sum() / ((outs[0] - nxt) ** 2).sum()).item()
                if snr < 35:
                    raise RuntimeError(f"block {i}: replay diverges from the next block's input ({snr:.1f} dB)")
            saved = {n: self.mapper.params[n].data.clone() for n, _ in keys}
            for n in saved:
                self.mapper.check(n)
            for (name, fmt), w in zip(keys, self._weights([(n, formats[f]) for n, f in keys])):
                p = self.mapper.params[name]
                p.data.copy_(torch.from_numpy(w))
                outs.append(self._run_block(i, x))
                p.data.copy_(saved[name])
        return keys, torch.cat(outs)

    @staticmethod
    def _block_errs(hidden):
        h2 = (hidden[0].double() ** 2).sum(-1)
        return [(((h - hidden[0]).double() ** 2).sum(-1) / h2).mean().item() for h in hidden[1:]]

    def costs(self, formats: dict[str, GGUFFile], progress=print) -> list[Cost]:
        """Exact cost of every quantizable matrix in every format."""
        names = formats[next(iter(formats))].quantized_matrices()
        result = []
        for i in range(len(self.blocks)):
            keys, hidden = self._block_variants(i, formats, names)
            kls = self._kl_from(i, hidden)
            result += [Cost(n, f, kl, be, formats[f].n_bytes(n))
                       for (n, f), kl, be in zip(keys, kls, self._block_errs(hidden))]
            progress(f"block {i:2d}: {len(keys)} variants, max KL {max(kls):.2e}")
        return result

    def output_gradients(self, n_samples, blocks, seed=0):
        """Gradients of log p(y) with respect to the output of the given blocks, y sampled from the model.

        For a small error e (all tokens, all channels) at the output of block i:
            KL ≈ ½ eᵀ F e,  F = E_y[g gᵀ],  g = ∇ Σ_t log p(y_t)
        (the Fisher matrix). Returns {i: [n_samples, T, d]} in BF16.
        """
        outs = {}

        def keep(i):
            def hook(module, args, out):
                outs[i] = out[0] if isinstance(out, tuple) else out
            return hook

        for p in self.model.parameters():
            p.requires_grad_(False)
        handles = [self.blocks[i].register_forward_hook(keep(i)) for i in blocks]
        self.norm.to(self.dtype)
        self.lm_head.to(self.dtype)
        try:
            with torch.enable_grad():
                # The weights require no gradient: requiring it on the input embeddings
                # builds the graph up to the block outputs.
                embeds = self.model.get_input_embeddings()(self.ids).detach().requires_grad_(True)
                logits = self.model(inputs_embeds=embeds, use_cache=False).logits[0].float()
                log_p = torch.log_softmax(logits, dim=-1)
                probs = log_p.detach().exp()
                gen = torch.Generator(device=logits.device).manual_seed(seed)  # same samples for every chunk of blocks
                targets = [outs[i] for i in blocks]
                grads = {i: [] for i in blocks}
                for _ in range(n_samples):
                    y = torch.multinomial(probs, 1, generator=gen)
                    g = torch.autograd.grad(log_p.gather(-1, y).sum(), targets, retain_graph=True)
                    for i, gi in zip(blocks, g):
                        grads[i].append(gi[0].to(torch.bfloat16))
        finally:
            for h in handles:
                h.remove()
            self.norm.float()
            self.lm_head.float()
        return {i: torch.stack(v) for i, v in grads.items()}

    def fisher_costs(self, formats: dict[str, GGUFFile], n_samples=128, grad_mem_gb=2.0, progress=print) -> list[Cost]:
        """Like costs(), with the KL approximated by ½ E_y[(g·e)²]. kl_se is the noise
        from the finite number of samples. Gradients are computed for chunks of blocks
        that fit in grad_mem_gb."""
        names = formats[next(iter(formats))].quantized_matrices()
        n_tokens, dim = self.calls[0][0][0].shape[-2:]
        step = max(1, int(grad_mem_gb * 2 ** 30 // (n_samples * n_tokens * dim * 2)))
        result = []
        for b0 in range(0, len(self.blocks), step):
            blocks = list(range(b0, min(len(self.blocks), b0 + step)))
            grads = self.output_gradients(n_samples, blocks)
            for i in blocks:
                keys, hidden = self._block_variants(i, formats, names)
                e = hidden[1:] - hidden[:1]  # [V, T, d]
                proj = torch.einsum("std,vtd->vs", grads[i].float(), e).double() ** 2  # (g·e)² per variant and sample
                kls = (0.5 * proj.mean(-1) / n_tokens).tolist()
                ses = (0.5 * proj.std(-1) / n_samples ** 0.5 / n_tokens).tolist()
                result += [Cost(n, f, kl, be, formats[f].n_bytes(n), se)
                           for (n, f), kl, be, se in zip(keys, kls, self._block_errs(hidden), ses)]
                progress(f"block {i:2d}: {len(keys)} variants (Fisher)")
            del grads
        return result

    def kl_of_map(self, choice: dict[str, str], formats: dict[str, GGUFFile]) -> float:
        """Output KL with every matrix of the map quantized at once (to check additivity)."""
        ref = self.calls[0][0][0]
        hidden = torch.cat([ref, ref])  # [reference, map]
        for i in range(len(self.blocks)):
            with self._float32(i):
                block_names = [n for n in choice if n.startswith(f"blk.{i}.")]
                saved = {n: self.mapper.params[n].data.clone() for n in block_names}
                for n in block_names:
                    self.mapper.check(n)
                ref_out = self._run_block(i, hidden[:1])
                for n in block_names:
                    p = self.mapper.params[n]
                    p.data.copy_(torch.from_numpy(self.mapper.to_hf(n, formats[choice[n]].weight(n, tuple(p.shape)))))
                map_out = self._run_block(i, hidden[1:])
                for n, w in saved.items():
                    self.mapper.params[n].data.copy_(w)
            hidden = torch.cat([ref_out, map_out])
        ref_lp, lp = self._log_probs(hidden[:1]), self._log_probs(hidden[1:])
        return (ref_lp.exp() * (ref_lp - lp)).sum(-1).mean().item()
