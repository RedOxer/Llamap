"""Pick one quantization format per tensor under a byte budget.

This is a multiple-choice knapsack: choose a format t_i for each tensor i to
minimise Σ cost[i][t_i] subject to Σ size[i][t_i] ≤ budget. Costs are assumed
additive across tensors, which makes the problem separable.

  1. Lagrangian relaxation: replace the constraint by a penalty λ per byte, so
     each tensor independently takes argmin_t cost[i][t] + λ·size[i][t].
     Bisection finds the smallest λ that fits the budget.
  2. Greedy fill: spend the remaining bytes on the upgrades with the best
     cost reduction per added byte.
"""

Table = dict[str, dict[str, float]]


def total(choice: dict[str, str], table: Table) -> float:
    return sum(table[name][fmt] for name, fmt in choice.items())


def pick(cost: Table, size: Table, lam: float) -> dict[str, str]:
    return {n: min(cost[n], key=lambda t: cost[n][t] + lam * size[n][t]) for n in cost}


def lagrange_multiplier(cost: Table, size: Table, budget: float, iterations: int = 100) -> float:
    """Smallest λ (by bisection) for which pick(λ) fits the budget."""
    lo, hi = 0.0, 1.0
    while total(pick(cost, size, hi), size) > budget:
        hi *= 2
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if total(pick(cost, size, mid), size) > budget:
            lo = mid
        else:
            hi = mid
    return hi


def solve(cost: Table, size: Table, budget: float, iterations: int = 100) -> dict[str, str]:
    """{tensor: format} of (approximately) minimal total cost under the budget.

    cost[i][t]: predicted cost of tensor i in format t.
    size[i][t]: size in bytes of tensor i in format t.
    """
    if cost.keys() != size.keys() or any(cost[n].keys() != size[n].keys() for n in cost):
        raise ValueError("cost and size must cover the same tensors and formats")
    smallest = {n: min(size[n], key=size[n].get) for n in size}
    if total(smallest, size) > budget:
        raise ValueError(f"budget too small: the smallest format everywhere takes {total(smallest, size):.0f} bytes")

    choice = pick(cost, size, lagrange_multiplier(cost, size, budget, iterations))

    while True:
        room = budget - total(choice, size)
        best = None  # (gain per byte, tensor, format)
        for n in cost:
            for t in cost[n]:
                extra = size[n][t] - size[n][choice[n]]
                gain = cost[n][choice[n]] - cost[n][t]
                if 0 < extra <= room and gain > 0 and (best is None or gain / extra > best[0]):
                    best = (gain / extra, n, t)
        if best is None:
            return choice
        choice[best[1]] = best[2]
