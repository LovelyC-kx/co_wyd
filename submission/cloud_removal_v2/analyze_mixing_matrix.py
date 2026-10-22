"""
Numerical computation of the second eigenvalue λ₂ of the mixing matrix
used by our Gossip aggregation on a chain-5 inter-plane topology.

Purpose: make §VI-D of the paper quantitatively precise about whether our
implementation places us on the Gossip-favored (τ(1−λ₂) >= 1) or
RelaySum-favored (τ(1−λ₂) < 1) side of the crossover.

Our `cloud_removal_v1/constellation.py:_gossip_average` does:
    participants = [p] + [q for q in chain neighbors of p]
    new_state_p = uniform_mean(participants)
which gives a row-stochastic but NOT doubly-stochastic W.

We compare three mixing matrices on chain-5:
 1. Ours (degree-uniform self+neighbor average, non-doubly-stochastic).
 2. Metropolis weights (standard gossip SGD setting, doubly stochastic).
 3. Uniform lazy random walk (Koloskova 2020 typical setting, double stochastic).

Run:   python -m cloud_removal_v2.analyze_mixing_matrix
"""

from __future__ import annotations

import numpy as np


def build_chain_gossip_ours(N: int) -> np.ndarray:
    """W as implemented in `_gossip_average` (uniform self+neighbors)."""
    W = np.zeros((N, N))
    for p in range(N):
        participants = [p]
        if p > 0:
            participants.append(p - 1)
        if p < N - 1:
            participants.append(p + 1)
        w = 1.0 / len(participants)
        for q in participants:
            W[p, q] = w
    return W


def build_metropolis(N: int) -> np.ndarray:
    """Metropolis weights on chain-N."""
    degs = [1 if p == 0 or p == N - 1 else 2 for p in range(N)]
    W = np.zeros((N, N))
    for p in range(N):
        for q in (p - 1, p + 1):
            if 0 <= q < N:
                W[p, q] = 1.0 / (1 + max(degs[p], degs[q]))
        W[p, p] = 1.0 - W[p, :].sum()
    return W


def build_uniform_lazy(N: int) -> np.ndarray:
    """Proper lazy random walk W = (I + D^-1 A) / 2 for irregular graph.
    (The simpler (I + A/d_max)/2 is NOT row-stochastic when the graph is
    irregular, so we do not use it.)"""
    A = np.zeros((N, N))
    for p in range(N - 1):
        A[p, p + 1] = 1.0
        A[p + 1, p] = 1.0
    degrees = A.sum(axis=1)
    D_inv = np.diag(1.0 / degrees)
    P = D_inv @ A
    return 0.5 * (np.eye(N) + P)


def analyze(name: str, W: np.ndarray, tau: int) -> dict:
    eigvals_complex = np.linalg.eigvals(W)
    eigvals = np.sort(np.abs(eigvals_complex))[::-1]
    lambda1 = eigvals[0]
    lambda2 = eigvals[1]
    spectral_gap = 1.0 - lambda2
    tau_times_gap = tau * spectral_gap
    tau2_times_gap = (tau ** 2) * spectral_gap
    print(f"\n=== {name} ===")
    print("W =")
    for row in W:
        print("  " + "  ".join(f"{v:.4f}" for v in row))
    print(f"Row-stochastic?  {np.allclose(W.sum(axis=1), 1.0)}")
    print(f"Col-stochastic?  {np.allclose(W.sum(axis=0), 1.0)}")
    print(f"Symmetric?       {np.allclose(W, W.T)}")
    print(f"|eigenvalues|    {['{:.4f}'.format(e) for e in eigvals]}")
    print(f"λ_1              {lambda1:.4f}")
    print(f"λ_2              {lambda2:.4f}")
    print(f"1 − λ_2          {spectral_gap:.4f}")
    print(f"τ = diameter     {tau}")
    print(f"τ(1 − λ_2)       {tau_times_gap:.4f}  {'RelaySum favored' if tau_times_gap < 1 else 'Gossip/AllReduce favored'}")
    print(f"τ²(1 − λ_2)²     {(tau_times_gap)**2:.4f}  (alt. crossover form)")
    return {
        "name": name,
        "lambda2": lambda2,
        "spectral_gap": spectral_gap,
        "tau_times_gap": tau_times_gap,
    }


def main() -> None:
    N = 5
    tau = N - 1
    print(f"Chain-{N} inter-plane topology, diameter τ = {tau}")
    print("Computing λ_2 for 3 mixing-matrix variants...")

    W_ours = build_chain_gossip_ours(N)
    W_metropolis = build_metropolis(N)
    W_lazy = build_uniform_lazy(N)

    results = []
    results.append(analyze("Ours (degree-uniform self+neighbor average)", W_ours, tau))
    results.append(analyze("Metropolis weights", W_metropolis, tau))
    results.append(analyze("Uniform lazy random walk", W_lazy, tau))

    print("\n" + "=" * 70)
    print("Summary for paper §VI-D:")
    print("=" * 70)
    for r in results:
        favored = "RelaySum" if r["tau_times_gap"] < 1 else "Gossip/AllReduce"
        print(f"  {r['name']:50s}  λ_2={r['lambda2']:.3f}  τ(1−λ_2)={r['tau_times_gap']:.3f}  [{favored}]")
    print()
    print("Interpretation:")
    print("  τ(1 − λ_2) vs 1 is the Vogels-2021-vs-Koloskova-2020 crossover heuristic")
    print("  for whether RelaySum's τ·σ² penalty is dominated by Gossip's ζ²/(1−λ_2) term.")
    print("  Smaller τ(1−λ_2) favors RelaySum; larger favors Gossip/AllReduce.")


if __name__ == "__main__":
    main()
