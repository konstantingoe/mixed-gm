"""Mini benchmark: polychoric solver comparison (Newton vs Brent).

Run with:
    python scribbling/random_stuff.py
"""

from __future__ import annotations

import time
import textwrap
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from hume.correlation import PolychoricCorrelation


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _make_ordinal_pair(
    n: int,
    rho: float,
    n_cats: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw two ordinal variables from a latent bivariate Gaussian.

    Categories are created by cutting the latent variables at equally-spaced
    quantiles, giving balanced marginals.

    Args:
        n: Sample size.
        rho: True latent correlation.
        n_cats: Number of ordinal categories (≥ 2).
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of two integer-coded ordinal arrays, each in {0, …, n_cats-1}.
    """
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n)
    cuts = np.linspace(0, 100, n_cats + 1)[1:-1]  # n_cats-1 interior percentiles
    x = np.searchsorted(np.percentile(z[:, 0], cuts), z[:, 0])
    y = np.searchsorted(np.percentile(z[:, 1], cuts), z[:, 1])
    return x.astype(float), y.astype(float)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    """Container for a single benchmark result."""

    solver: str
    n: int
    rho_true: float
    n_cats: int
    rho_hat: float
    elapsed_s: float
    reps: int
    bias: float = field(init=False)
    elapsed_ms: float = field(init=False)

    def __post_init__(self) -> None:
        """Compute bias and per-fit time in milliseconds."""
        self.bias = self.rho_hat - self.rho_true
        self.elapsed_ms = self.elapsed_s / self.reps * 1000


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    sample_sizes: list[int],
    rhos: list[float],
    category_counts: list[int],
    reps: int = 5,
    seed: int = 42,
) -> list[BenchResult]:
    """Run all (n, rho, n_cats) × solver combinations.

    Each configuration is timed over *reps* repetitions on the **same**
    dataset so that solver-iteration behaviour — not data-generation noise —
    dominates the measurement.

    Args:
        sample_sizes: List of sample sizes to benchmark.
        rhos: List of true latent correlations.
        category_counts: List of ordinal category counts.
        reps: Number of timed repetitions per configuration.
        seed: Master seed; each (n, rho, n_cats) cell gets a derived seed.

    Returns:
        List of :class:`BenchResult` objects, one per (solver, n, rho, n_cats).
    """
    results: list[BenchResult] = []

    configs = list(product(sample_sizes, rhos, category_counts))
    _ = len(configs)

    for idx, (n, rho, n_cats) in enumerate(configs, start=1):
        cell_seed = seed + idx * 1000
        x, y = _make_ordinal_pair(n=n, rho=rho, n_cats=n_cats, seed=cell_seed)

        for solver in ("newton", "brent"):
            est = PolychoricCorrelation(solver=solver)

            # Warm-up (not timed)
            est.fit(x, y)
            rho_hat = est.correlation

            # Timed repetitions
            t0 = time.perf_counter()
            for _ in range(reps):
                PolychoricCorrelation(solver=solver).fit(x, y)
            elapsed = time.perf_counter() - t0

            results.append(
                BenchResult(
                    solver=solver,
                    n=n,
                    rho_true=rho,
                    n_cats=n_cats,
                    rho_hat=rho_hat,
                    elapsed_s=elapsed,
                    reps=reps,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------


def print_table(results: list[BenchResult]) -> None:
    """Print a formatted comparison table grouped by configuration."""
    header = f"{'n':>7}  {'ρ_true':>6}  {'cats':>4}  {'solver':>6}  {'ρ̂':>7}  {'bias':>8}  {'ms/fit':>8}"
    sep = "-" * len(header)

    print("\nPolychoric solver benchmark")
    print(sep)
    print(header)
    print(sep)

    prev_key: tuple[int, float, int] | None = None
    for r in results:
        key = (r.n, r.rho_true, r.n_cats)
        if prev_key is not None and key != prev_key:
            print()
        prev_key = key

        print(
            f"{r.n:>7}  {r.rho_true:>6.2f}  {r.n_cats:>4}  "
            f"{r.solver:>6}  {r.rho_hat:>7.4f}  {r.bias:>+8.4f}  {r.elapsed_ms:>8.2f}"
        )

    print(sep)

    # Summary: mean speedup of newton over brent per config
    newton_map = {(r.n, r.rho_true, r.n_cats): r.elapsed_ms for r in results if r.solver == "newton"}
    brent_map = {(r.n, r.rho_true, r.n_cats): r.elapsed_ms for r in results if r.solver == "brent"}
    speedups = [brent_map[k] / newton_map[k] for k in newton_map if k in brent_map]
    if speedups:
        print(
            f"\nNewton speedup over Brent — "
            f"mean: {np.mean(speedups):.2f}×  "
            f"min: {np.min(speedups):.2f}×  "
            f"max: {np.max(speedups):.2f}×"
        )

    # Accuracy: max |bias| per solver
    for solver in ("newton", "brent"):
        biases = [abs(r.bias) for r in results if r.solver == solver]
        print(f"Max |bias| ({solver:>6}): {max(biases):.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print(
        textwrap.dedent("""\
        Configuration
        -------------
        Solvers : newton (Fisher scoring), brent (minimize_scalar bounded)
        Reps    : 5 timed fits per cell on identical data
        Data    : balanced ordinal categories from latent bivariate Gaussian
        """)
    )

    results = run_benchmark(
        sample_sizes=[200, 1_000, 5_000],
        rhos=[0.0, 0.3, 0.7],
        category_counts=[2, 4, 7],
        reps=5,
    )

    print_table(results)
