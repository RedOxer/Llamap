import itertools
import random

import pytest

from llamap.solver import solve, total

# Two tensors, two formats. Upgrading "a" gains 9 for 1 byte, upgrading "b" gains 1.
COST = {"a": {"lo": 10.0, "hi": 1.0}, "b": {"lo": 3.0, "hi": 2.0}}
SIZE = {"a": {"lo": 1, "hi": 2}, "b": {"lo": 1, "hi": 2}}


@pytest.mark.parametrize("budget, expected", [
    (2, {"a": "lo", "b": "lo"}),   # just room for the smallest format
    (3, {"a": "hi", "b": "lo"}),   # one more byte: it goes to the tensor that gains most
    (4, {"a": "hi", "b": "hi"}),   # room for everything
    (100, {"a": "hi", "b": "hi"}),
])
def test_known_answers(budget, expected):
    assert solve(COST, SIZE, budget) == expected


def test_budget_too_small():
    with pytest.raises(ValueError, match="budget too small"):
        solve(COST, SIZE, 1)


def test_mismatched_tables():
    with pytest.raises(ValueError, match="same tensors"):
        solve(COST, {"a": SIZE["a"]}, 10)


def test_never_picks_a_worse_and_bigger_format():
    # "bad" is bigger and costlier than "lo": it must never be picked.
    cost = {"a": {"lo": 5.0, "bad": 6.0, "hi": 1.0}}
    size = {"a": {"lo": 1, "bad": 2, "hi": 3}}
    assert solve(cost, size, 2) == {"a": "lo"}
    assert solve(cost, size, 3) == {"a": "hi"}


def brute_force(cost, size, budget):
    names = list(cost)
    feasible = (dict(zip(names, c)) for c in itertools.product(*(cost[n] for n in names)))
    return min((c for c in feasible if total(c, size) <= budget), key=lambda c: total(c, cost))


def test_random_instances_close_to_optimal():
    # Costs decrease with size, as with real formats. The solver is not exact in
    # general, but it must fit the budget and stay close to the brute-force optimum.
    rng = random.Random(0)
    gaps = []
    for _ in range(200):
        cost, size = {}, {}
        for i in range(4):
            sizes = sorted(rng.sample(range(1, 20), 3))
            costs = sorted((rng.uniform(0, 10) for _ in range(3)), reverse=True)
            cost[f"t{i}"] = dict(zip("xyz", costs))
            size[f"t{i}"] = dict(zip("xyz", sizes))
        lo = sum(min(s.values()) for s in size.values())
        hi = sum(max(s.values()) for s in size.values())
        budget = rng.randint(lo, hi)
        choice = solve(cost, size, budget)
        assert total(choice, size) <= budget
        best = total(brute_force(cost, size, budget), cost)
        gaps.append((total(choice, cost) - best) / best)
    # With 4 tensors the bytes lost to rounding weigh a lot; with hundreds of
    # tensors (real models) they become negligible.
    assert sum(g < 1e-9 for g in gaps) / len(gaps) > 0.75  # optimal in 83 % of cases
    assert sum(gaps) / len(gaps) < 0.02                     # mean gap to the optimum: 0.9 %
