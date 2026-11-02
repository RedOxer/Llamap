import numpy as np

from llamap.hf_gguf import permute, unpermute


def test_unpermute_inverts_permute():
    rng = np.random.default_rng(0)
    for n_head, head_dim in [(32, 128), (8, 128), (4, 64)]:
        w = rng.standard_normal((n_head * head_dim, 16)).astype(np.float32)
        assert np.array_equal(unpermute(permute(w, n_head), n_head), w)


def test_permute_interleaves_halves_of_each_head():
    # One head of dimension 4: rows [a0 a1 b0 b1] -> [a0 b0 a1 b1].
    w = np.arange(4).reshape(4, 1)
    assert permute(w, 1).ravel().tolist() == [0, 2, 1, 3]
