"""Just some testing code for random stuff."""

from collections.abc import Callable

import numpy as np
from hume.correlation import thresholds
from scipy.optimize import brentq
from scipy import stats


def pi_rs(
    lower: tuple[float, float],
    upper: tuple[float, float],
    corr: float,
) -> float:
    """Calculate rectangle probabilities.

    Args:§
        lower (tuple[float, float]): _description_
        upper (tuple[float, float]  ): _description_
        corr (float): _description_

    Returns:
        float: _description_
    """
    mvn = stats.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[1.0, corr], [corr, 1.0]]))  # pyright: ignore[reportArgumentType]

    l1, l2 = lower
    u1, u2 = upper

    cdf_u1_u2 = mvn.cdf([u1, u2])
    cdf_l1_u2 = mvn.cdf([l1, u2])
    cdf_u1_l2 = mvn.cdf([u1, l2])
    cdf_l1_l2 = mvn.cdf([l1, l2])

    return float(cdf_u1_u2 - cdf_l1_u2 - cdf_u1_l2 + cdf_l1_l2)


def safe_pdf(mvn: stats._multivariate.multivariate_normal_frozen, x: np.ndarray) -> float:
    """Safely evaluate bivariate standard Gaussian pdf.

    Args:
        mvn (stats._multivariate.multivariate_normal_frozen): _description_
        x (np.ndarray): _description_

    Returns:
        float: _description_
    """
    return 0.0 if np.any(np.isinf(x)) else mvn.pdf(x)


def pi_rs_derivative(lower: np.ndarray, upper: np.ndarray, corr: float) -> float:
    """Derivative of pi_rs.

    Args:
        lower (np.ndarray): _description_
        upper (np.ndarray): _description_
        corr (float): _description_

    Returns:
        float: _description_
    """
    mvn = stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, corr], [corr, 1.0]])  # pyright: ignore[reportArgumentType]
    l1, l2 = lower
    u1, u2 = upper

    return (
        safe_pdf(mvn, np.array([u1, u2]))
        - safe_pdf(mvn, np.array([l1, u2]))
        - safe_pdf(mvn, np.array([u1, l2]))
        + safe_pdf(mvn, np.array([l1, l2]))
    )


def find_feasible_interval(
    score: Callable[[float], float], lo: float = -0.9999, hi: float = 0.9999, max_iter: int = 20
) -> tuple[float, float]:
    """Find feasible interval for brentq.

    Args:
        score (_type_): _description_
        lo (float, optional): _description_. Defaults to -0.9999.
        hi (float, optional): _description_. Defaults to 0.9999.
        max_iter (int, optional): _description_. Defaults to 20.

    Raises:
        RuntimeError: _description_

    Returns:
        tuple[float, float]: _description_
    """
    for _ in range(max_iter):
        try:
            score(lo)
            score(hi)
            return lo, hi
        except ZeroDivisionError:
            lo *= 0.9
            hi *= 0.9
    raise RuntimeError("Could not find feasible correlation interval")


def polychoric_correlation(x: np.ndarray, y: np.ndarray, max_cor: float = 0.9999) -> float:
    """Compute polychoric correlation estimate between two discrete variables.

    Args:
        x (np.ndarray): _description_
        y (np.ndarray): _description_
        max_cor (float, optional): _description_. Defaults to 0.9999.

    Raises:
        ZeroDivisionError: _description_

    Returns:
        float: _description_
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = x.size

    ux = np.unique(x)
    uy = np.unique(y)

    # Joint frequency table
    n_rs = np.array(
        [[np.sum((x == xi) & (y == yj)) for yj in uy] for xi in ux],
        dtype=float,
    )
    assert n_rs.sum() == n

    tx = thresholds(x)
    ty = thresholds(y)

    def cell_term(i: int, j: int, corr: float) -> float:
        lower = (float(tx[i]), float(ty[j]))
        upper = (float(tx[i + 1]), float(ty[j + 1]))

        p = pi_rs(lower=lower, upper=upper, corr=corr)

        if p < 1e-12:
            if n_rs[i, j] == 0:
                return 0.0
            raise ZeroDivisionError(f"π_rs ≈ 0 for nonempty cell ({i}, {j})")

        dp = pi_rs_derivative(lower=np.array(lower), upper=np.array(upper), corr=corr)

        # I think currently this is incorrect since the n_rs
        #  are not necessarily ordered according to the enumeration of thresholds.
        return float(n_rs[i, j] * dp / p)

    def score(corr: float) -> float:
        """Calculate score function.

        Args:
            corr (float): _description_

        Returns:
            float: _description_
        """
        return sum(cell_term(i, j, corr) for i in range(len(ux)) for j in range(len(uy)))

    a, b = find_feasible_interval(score)

    root = brentq(score, a, b)
    if isinstance(root, tuple):
        root = root[0]

    return float(np.clip(float(root), -max_cor, max_cor))


if main := __name__ == "__main__":
    rng = np.random.default_rng()

    z_data = rng.multivariate_normal(
        mean=np.zeros(2),
        cov=np.array([
            [
                1,
                0.4,
            ],
            [0.4, 1],
        ]),
        size=10000,
    )

    np.corrcoef(z_data, rowvar=False)

    x = (z_data[:, 0] > 0).astype(int)
    y = (z_data[:, 1] > 0).astype(int)

    print(polychoric_correlation(x, y))
