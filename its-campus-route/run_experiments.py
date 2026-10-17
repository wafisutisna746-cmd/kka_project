import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIZES = [10, 20, 30, 40]
HK_MAX = 16
RUNS = 10
SA_ITERS = 20000
GA_POP = 100
GA_GENS = 200
MA_POP = 30
MA_GENS = 100

Path("results").mkdir(exist_ok=True)
D_FULL = np.load("data/dist_matrix.npy")
NAMES = pd.read_csv("data/stops.csv")["name"].tolist()


def tour_cost(t, D):
    return sum(D[t[i], t[(i + 1) % len(t)]] for i in range(len(t)))


def held_karp(D):
    n = len(D)
    inf = float("inf")
    dp = [[inf] * n for _ in range(1 << n)]
    dp[1][0] = 0.0
    for mask in range(1, 1 << n):
        if not mask & 1:
            continue
        for j in range(n):
            cur = dp[mask][j]
            if cur == inf or not (mask >> j) & 1:
                continue
            for k in range(1, n):
                if (mask >> k) & 1:
                    continue
                nm = mask | (1 << k)
                v = cur + D[j][k]
                if v < dp[nm][k]:
                    dp[nm][k] = v
    full = (1 << n) - 1
    return min(dp[full][j] + D[j][0] for j in range(1, n))


def two_opt(t, D):
    n = len(t)
    t = t[:]
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = t[i], t[i + 1]
                c, d = t[j], t[(j + 1) % n]
                if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-9:
                    t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                    improved = True
    return t


def nearest_neighbor(D, start=0):
    n = len(D)
    tour = [start]
    left = set(range(n)) - {start}
    while left:
        nxt = min(left, key=lambda j: D[tour[-1], j])
        tour.append(nxt)
        left.remove(nxt)
    c = tour_cost(tour, D)
    return tour, c, [c]


def simulated_annealing(D, rng, iters=SA_ITERS, t_end=1e-2):
    n = len(D)
    cur = list(range(n))
    rng.shuffle(cur)
    cc = tour_cost(cur, D)
    best, bc = cur[:], cc
    t0 = 0.05 * cc
    alpha = (t_end / t0) ** (1.0 / iters)
    temp = t0
    hist = []
    for _ in range(iters):
        i, j = sorted(rng.sample(range(n), 2))
        cand = cur[:i] + cur[i:j + 1][::-1] + cur[j + 1:]
        nc = tour_cost(cand, D)
        delta = nc - cc
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            cur, cc = cand, nc
            if cc < bc:
                best, bc = cur[:], cc
        hist.append(bc)
        temp *= alpha
    return best, bc, hist


def order_crossover(p1, p2, rng):
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))
    child = [None] * n
    child[a:b + 1] = p1[a:b + 1]
    used = set(child[a:b + 1])
    fill = [g for g in p2 if g not in used]
    pos = list(range(0, a)) + list(range(b + 1, n))
    for p, g in zip(pos, fill):
        child[p] = g
    return child


def mutate(t, rng):
    t = t[:]
    i, j = sorted(rng.sample(range(len(t)), 2))
    if rng.random() < 0.5:
        t[i], t[j] = t[j], t[i]
    else:
        t[i:j + 1] = t[i:j + 1][::-1]
    return t


def genetic_algorithm(D, rng, pop_size=GA_POP, gens=GA_GENS, memetic=False,
                      cx_rate=0.9, mut_rate=0.2, elite=2, k=3):
    n = len(D)
    pop = []
    for _ in range(pop_size):
        t = list(range(n))
        rng.shuffle(t)
        pop.append(two_opt(t, D) if memetic else t)
    costs = [tour_cost(t, D) for t in pop]
    hist = []
    for _ in range(gens):
        order = sorted(range(pop_size), key=lambda i: costs[i])
        new_pop = [pop[i] for i in order[:elite]]
        while len(new_pop) < pop_size:
            p1 = min(rng.sample(range(pop_size), k), key=lambda i: costs[i])
            p2 = min(rng.sample(range(pop_size), k), key=lambda i: costs[i])
            child = order_crossover(pop[p1], pop[p2], rng) if rng.random() < cx_rate else pop[p1][:]
            if rng.random() < mut_rate:
                child = mutate(child, rng)
            if memetic:
                child = two_opt(child, D)
            new_pop.append(child)
        pop = new_pop
        costs = [tour_cost(t, D) for t in pop]
        hist.append(min(costs))
    bi = int(np.argmin(costs))
    best_hist = np.minimum.accumulate(hist).tolist()
    return pop[bi], costs[bi], best_hist


ALGOS = {
    "NearestNeighbor": lambda D, rng: nearest_neighbor(D),
    "SimulatedAnnealing": lambda D, rng: simulated_annealing(D, rng),
    "GA": lambda D, rng: genetic_algorithm(D, rng),
    "GA+2opt": lambda D, rng: genetic_algorithm(D, rng, MA_POP, MA_GENS, memetic=True),
}
EVAL_PER_STEP = {"SimulatedAnnealing": 1, "GA": GA_POP, "GA+2opt": MA_POP}

rows = []
conv = {}
exact_flag = {}
sizes = sorted({min(x, len(D_FULL)) for x in SIZES})
for n in sizes:
    idx = sorted(random.Random(0).sample(range(len(D_FULL)), n))
    D = D_FULL[np.ix_(idx, idx)]
    opt = held_karp(D.tolist()) if n <= HK_MAX else None
    exact_flag[n] = opt is not None
    start = len(rows)
    for name, fn in ALGOS.items():
        runs = 1 if name == "NearestNeighbor" else RUNS
        hists = []
        for r in range(runs):
            rng = random.Random(r)
            t0 = time.perf_counter()
            tour, cost, hist = fn(D, rng)
            dt = time.perf_counter() - t0
            rows.append({
                "n": n, "algo": name, "run": r, "cost": cost, "time_s": dt,
                "tour": [NAMES[idx[i]] for i in tour],
            })
            hists.append(hist)
        conv[(n, name)] = np.mean(hists, axis=0)
    ref = opt if opt is not None else min(x["cost"] for x in rows[start:])
    for x in rows[start:]:
        x["optimal"] = ref
        x["gap_pct"] = max(0.0, 100 * (x["cost"] - ref) / ref)
    print(f"n={n} reference={ref:.1f} ({'exact' if opt is not None else 'best found'})")

res = pd.DataFrame(rows)
res.to_csv("results/results.csv", index=False)
summary = res.groupby(["n", "algo"]).agg(
    mean_cost=("cost", "mean"), std_cost=("cost", "std"), best_cost=("cost", "min"),
    mean_gap_pct=("gap_pct", "mean"), mean_time_s=("time_s", "mean"), optimal=("optimal", "first"),
).round(2)
summary.to_csv("results/summary.csv")
print(summary)

nmax = max(k[0] for k in conv)
fig, ax = plt.subplots(figsize=(8, 5))
for name in EVAL_PER_STEP:
    h = conv[(nmax, name)]
    ax.plot(np.arange(1, len(h) + 1) * EVAL_PER_STEP[name], h, label=name)
ax.axhline(res[res["n"] == nmax]["optimal"].iloc[0], color="k", ls="--",
           label="Optimal (Held-Karp)" if exact_flag[nmax] else "Best found")
ax.set_xscale("log")
ax.set_xlabel("Tour cost evaluations")
ax.set_ylabel("Best tour length (m)")
ax.set_title(f"Convergence, {nmax} stops (mean of {RUNS} runs)")
ax.legend()
fig.savefig("results/convergence.png", dpi=200, bbox_inches="tight")

best = res[(res["n"] == nmax)].sort_values("cost").iloc[0]
print("best tour found:", best["algo"], round(best["cost"], 1), "m")
print(" -> ".join(best["tour"]))
