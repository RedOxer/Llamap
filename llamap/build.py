"""Build the final GGUF from a {matrix: format} map.

Block matrices take the map's format; token_embd and output take the reference's
type, so that a comparison with the reference only involves the block matrices.
The output file is then checked.
"""
import re

from llamap.hf_gguf import GGUFFile
from llamap.llamacpp import LlamaCpp


def write_tensor_types(choice: dict[str, str], path):
    """The --tensor-type-file: one "^name$=format" line per matrix."""
    with open(path, "w") as f:
        for name in sorted(choice):
            f.write("^" + re.escape(name) + "$=" + choice[name] + "\n")


def build(llama: LlamaCpp, bf16, imatrix, choice: dict[str, str], ref: GGUFFile, map_path, out) -> GGUFFile:
    write_tensor_types(choice, map_path)
    llama.quantize(bf16, out, "Q4_K_M", imatrix, tensor_types=map_path,  # preset unused: the map covers every matrix
                   token_embd=ref.type_of("token_embd.weight"),
                   output=ref.type_of("output.weight") if "output.weight" in ref.tensors else None)
    built = GGUFFile(out)
    wrong = [n for n, t in choice.items() if built.type_of(n) != t]
    if wrong:
        raise RuntimeError(f"{len(wrong)} matrices do not have the requested format, e.g. {wrong[0]}")
    others = [n for n in ref.tensors if n not in choice]
    differ = [n for n in others if n not in built.tensors or built.type_of(n) != ref.type_of(n)]
    if differ:
        raise RuntimeError(f"tensors outside the map differ from the reference: {differ[:3]}")
    return built
