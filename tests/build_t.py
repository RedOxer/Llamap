import re

from llamap.build import write_tensor_types
from llamap.pipeline import append_costs, load_costs
from llamap.predict import Cost


def test_tensor_types_file_matches_exact_names(tmp_path):
    path = tmp_path / "map.txt"
    write_tensor_types({"blk.1.attn_k.weight": "q5_k", "blk.10.attn_k.weight": "q8_0"}, path)
    lines = path.read_text().splitlines()
    assert lines == [r"^blk\.1\.attn_k\.weight$=q5_k", r"^blk\.10\.attn_k\.weight$=q8_0"]
    # "blk.1" must not match "blk.10", nor "blkX1" (the dot is escaped).
    pattern = lines[0].split("=")[0]
    assert re.search(pattern, "blk.1.attn_k.weight")
    assert not re.search(pattern, "blk.10.attn_k.weight")
    assert not re.search(pattern, "blkX1.attn_k.weight")


def test_costs_csv_round_trip(tmp_path):
    path = tmp_path / "costs.csv"
    a = [Cost("blk.0.ffn_up.weight", "q4_k", 1e-4, 2e-6, 1000)]
    b = [Cost("blk.0.ffn_up.weight", "q6_k", 1e-5, 3e-7, 1500)]
    append_costs(path, a)
    append_costs(path, b)  # resuming: a new format is appended without a second header
    assert load_costs(path) == a + b
