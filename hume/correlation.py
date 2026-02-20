"""Correlation estimation for mixed continuous and ordinal data.

This module provides two correlation estimators for the latent Gaussian copula
model, plus a convenience dispatcher:

* :class:`PolychoricCorrelation` — maximum-likelihood polychoric correlation
  between two ordinal variables.
* :class:`PolyserialCorrelation` — ad-hoc polyserial correlation between one
  continuous and one ordinal variable.
* :func:`adhoc_polyserial` — module-level function that dispatches to the
  appropriate estimator based on the number of unique values of each variable.

References:
    Liu, Han, Lafferty, John and Wasserman, Larry. (2009).
    The nonparanormal: Semi-parametric estimation of high dimensional
    undirected graphs. Journal of Machine Learning Research 10(80), 2295–2328.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats
from collections.abc import Callable
from scipy.optimize import brentq

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Private utility functions
# ---------------------------------------------------------------------------


def _to_array(x: object) -> np.ndarray:
    """Convert *x* to a 1-D float64 NumPy array."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1-D array, got shape {arr.shape}.")
    return arr


def _validate_pair(x: np.ndarray, y: np.ndarray, min_samples: int = 5) -> None:
    """Shared sanity checks for a pair of arrays.

    Raises:
        ValueError: If any structural invariant is violated.
    """
    if x.size == 0 or y.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if x.size != y.size:
        raise ValueError(f"Input arrays must have the same length, got {x.size} and {y.size}.")
    if x.size < min_samples:
        raise ValueError(f"At least {min_samples} observations are required, got {x.size}.")
    if np.all(np.isnan(x)) or np.all(np.isnan(y)):
        raise ValueError("Input array is entirely NaN.")
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        raise ValueError("Input array is constant — correlation is undefined.")


def _validate_ordinal(arr: np.ndarray, name: str = "array") -> None:
    """Extra checks specific to an ordinal variable.

    Raises:
        ValueError: If the array does not look like a valid ordinal variable.
    """
    if np.any(np.isinf(arr)):
        raise ValueError(f"Ordinal variable '{name}' contains infinite values.")
    n_unique = len(np.unique(arr[~np.isnan(arr)]))
    if n_unique < 2:
        raise ValueError(f"Ordinal variable '{name}' must have at least 2 distinct categories, found {n_unique}.")


def _validate_continuous(arr: np.ndarray, name: str = "array") -> None:
    """Extra checks for a continuous variable.

    Raises:
        ValueError: If the array contains non-finite values that would break
            the nonparanormal transformation.
    """
    if np.any(np.isinf(arr)):
        raise ValueError(
            f"Continuous variable '{name}' contains infinite values. Please remove or replace them before fitting."
        )
    n_nan = int(np.sum(np.isnan(arr)))
    if n_nan > 0:
        logger.warning(
            "Continuous variable '%s' contains %d NaN value(s); they will be excluded from the rank transform.",
            name,
            n_nan,
        )


def _thresholds(disc_var: np.ndarray) -> np.ndarray:
    """Compute normal-score thresholds for a discrete/ordinal variable."""
    n = len(disc_var)
    _, counts = np.unique(disc_var, return_counts=True)
    cumm_proportions = np.concatenate([[0], np.cumsum(counts / n)])
    return stats.norm.ppf(cumm_proportions)


def _f_hat(x: np.ndarray) -> np.ndarray:
    """Winsorized nonparanormal transformation (Liu et al., 2009)."""
    n = x.shape[0]
    npn_thresh = 1 / (4 * (n**0.25) * np.sqrt(np.pi * np.log(n)))
    ranks = stats.rankdata(x, method="average")
    ecdf_values = ranks / (n + 1)
    ecdf_winsorized = np.clip(ecdf_values, npn_thresh, 1 - npn_thresh)
    transformed = stats.norm.ppf(ecdf_winsorized)
    transformed /= np.std(transformed, ddof=1)
    return transformed


def _npn_pearson(cont: np.ndarray, disc: np.ndarray) -> float:
    """Nonparanormal Pearson correlation (private helper)."""
    return float(np.corrcoef(_f_hat(cont), disc)[0, 1])


def _pi_rs(
    lower: tuple[float, float],
    upper: tuple[float, float],
    corr: float,
) -> float:
    """Bivariate normal rectangle probability P(l1 < X ≤ u1, l2 < Y ≤ u2)."""
    mvn = stats.multivariate_normal(  # pyright: ignore[reportArgumentType]
        mean=np.array([0.0, 0.0]),
        cov=np.array([[1.0, corr], [corr, 1.0]]),
    )
    l1, l2 = lower
    u1, u2 = upper
    return float(mvn.cdf([u1, u2]) - mvn.cdf([l1, u2]) - mvn.cdf([u1, l2]) + mvn.cdf([l1, l2]))


def _safe_mvn_pdf(mvn: stats._multivariate.multivariate_normal_frozen, x: np.ndarray) -> float:
    """Evaluate bivariate normal pdf, returning 0 for infinite arguments."""
    return 0.0 if np.any(np.isinf(x)) else float(mvn.pdf(x))


def _pi_rs_derivative(lower: np.ndarray, upper: np.ndarray, corr: float) -> float:
    """Score-function derivative of _pi_rs w.r.t. *corr*."""
    mvn = stats.multivariate_normal(  # pyright: ignore[reportArgumentType]
        mean=[0.0, 0.0], cov=[[1.0, corr], [corr, 1.0]]
    )
    l1, l2 = lower
    u1, u2 = upper
    return (
        _safe_mvn_pdf(mvn, np.array([u1, u2]))
        - _safe_mvn_pdf(mvn, np.array([l1, u2]))
        - _safe_mvn_pdf(mvn, np.array([u1, l2]))
        + _safe_mvn_pdf(mvn, np.array([l1, l2]))
    )


def _find_feasible_interval(
    score: Callable[[float], float],
    lo: float = -0.9999,
    hi: float = 0.9999,
    max_iter: int = 20,
) -> tuple[float, float]:
    """Shrink the search interval until *score* evaluates without error at both ends."""
    for _ in range(max_iter):
        try:
            score(lo)
            score(hi)
            return lo, hi
        except ZeroDivisionError:
            lo *= 0.9
            hi *= 0.9
    raise RuntimeError(
        "Could not find a feasible correlation interval for brentq. "
        "Check that the ordinal variables have sufficient overlap."
    )


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class CorrelationMeasure(ABC):
    """Abstract base class for latent Gaussian copula correlation estimators.

    Subclasses must implement :meth:`fit`.  After fitting, the estimated
    correlation coefficient is available through the :attr:`correlation`
    property.

    Args:
        max_cor: Absolute maximum allowed correlation value.  Estimates that
            exceed this value in magnitude are clipped to ±*max_cor*.
            Defaults to 0.9999.
    """

    def __init__(self, max_cor: float = 0.9999) -> None:
        """Inits.

        Args:
            max_cor (float, optional): Defaults to 0.9999.

        Raises:
            ValueError: If *max_cor* is not in (0, 1).
        """
        if not (0 < max_cor < 1):
            raise ValueError(f"`max_cor` must be in (0, 1), got {max_cor}.")
        self._max_cor = max_cor
        self._correlation: float | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> CorrelationMeasure:
        """Compute the correlation estimate from data.

        Args:
            x: First variable.
            y: Second variable (must have the same length as *x*).

        Returns:
            *self* — enables method chaining.
        """

    @property
    def correlation(self) -> float:
        """Estimated correlation coefficient in [-max_cor, max_cor].

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        if self._correlation is None:
            raise RuntimeError("Call .fit(x, y) before accessing .correlation.")
        return self._correlation

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare(x: object, y: object) -> tuple[np.ndarray, np.ndarray]:
        """Convert inputs to float arrays and run shared sanity checks."""
        x_arr = _to_array(x)
        y_arr = _to_array(y)
        _validate_pair(x_arr, y_arr)
        return x_arr, y_arr

    def _clip(self, value: float) -> float:
        """Clip *value* to ±max_cor."""
        return float(np.clip(value, -self._max_cor, self._max_cor))


# ---------------------------------------------------------------------------
# Polychoric correlation
# ---------------------------------------------------------------------------


class PolychoricCorrelation(CorrelationMeasure):
    """Maximum-likelihood polychoric correlation between two ordinal variables.

    Both variables are treated as discretised versions of latent nonparanormal variates.
    The correlation of those latent variates is estimated by
    maximising the multinomial log-likelihood via Brent's root-finding method
    applied to the score equation.

    Args:
        max_cor: Absolute upper bound for the estimated correlation.
            Defaults to 0.9999.

    Example::

        rho = PolychoricCorrelation().fit(x_ord, y_ord).correlation
    """

    def fit(self, x: np.ndarray, y: np.ndarray) -> PolychoricCorrelation:
        """Fit the polychoric correlation model.

        Args:
            x: First ordinal variable (integer-valued or small number of
                distinct float levels).
            y: Second ordinal variable.

        Returns:
            *self*

        Raises:
            ValueError: On invalid or incompatible inputs.
            RuntimeError: If the root-finding algorithm cannot converge.
        """
        x_arr, y_arr = self._prepare(x, y)
        _validate_ordinal(x_arr, "x")
        _validate_ordinal(y_arr, "y")

        self._correlation = self._clip(self._polychoric(x_arr, y_arr))
        return self

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _polychoric(self, x: np.ndarray, y: np.ndarray) -> float:
        n = x.size
        ux = np.unique(x)
        uy = np.unique(y)

        n_rs = np.array(
            [[np.sum((x == xi) & (y == yj)) for yj in uy] for xi in ux],
            dtype=float,
        )
        assert n_rs.sum() == n, "Joint frequency table does not sum to n."

        tx = _thresholds(x)
        ty = _thresholds(y)

        def _cell_term(i: int, j: int, corr: float) -> float:
            lower = (float(tx[i]), float(ty[j]))
            upper = (float(tx[i + 1]), float(ty[j + 1]))
            p = _pi_rs(lower=lower, upper=upper, corr=corr)
            if p < 1e-12:
                if n_rs[i, j] == 0:
                    return 0.0
                raise ZeroDivisionError(
                    f"Bivariate cell probability ≈ 0 for non-empty cell ({i}, {j}). "
                    "Consider collapsing sparse categories."
                )
            dp = _pi_rs_derivative(lower=np.array(lower), upper=np.array(upper), corr=corr)
            return float(n_rs[i, j] * dp / p)

        def score(corr: float) -> float:
            return sum(_cell_term(i, j, corr) for i in range(len(ux)) for j in range(len(uy)))

        a, b = _find_feasible_interval(score)
        root = brentq(score, a, b)
        return float(root) if not isinstance(root, tuple) else float(root[0])


# ---------------------------------------------------------------------------
# Polyserial correlation
# ---------------------------------------------------------------------------


class PolyserialCorrelation(CorrelationMeasure):
    """Ad-hoc polyserial correlation between one continuous and one ordinal variable.

    The estimator applies the nonparanormal (rank-based) transformation to the
    continuous variable and uses a closed-form correction factor derived from
    the ordinal thresholds to approximate the underlying latent correlation.

    Args:
        max_cor: Absolute upper bound for the estimated correlation.
            Defaults to 0.9999.
        n_levels_threshold: Variables with fewer unique values than this
            threshold are treated as ordinal.  Defaults to 20.

    Example::

        rho = PolyserialCorrelation().fit(x_cont, y_ord).correlation
    """

    def __init__(
        self,
        max_cor: float = 0.9999,
        n_levels_threshold: int = 20,
    ) -> None:
        """Inits polyserial correlation class.

        Args:
            max_cor (float, optional): Defaults to 0.9999.
            n_levels_threshold (int, optional): Until what level set to treat a variable as ordinal. Defaults to 20.

        Raises:
            ValueError: _description_
        """
        super().__init__(max_cor=max_cor)
        if n_levels_threshold < 2:
            raise ValueError("`n_levels_threshold` must be ≥ 2.")
        self._n_levels_threshold = n_levels_threshold

    def fit(self, x: np.ndarray, y: np.ndarray) -> PolyserialCorrelation:
        """Fit the polyserial correlation model.

        The method automatically identifies which argument is the continuous
        variable and which is the ordinal variable based on the number of
        unique values.

        Args:
            x: One variable (continuous or ordinal).
            y: The other variable (ordinal or continuous).

        Returns:
            *self*

        Raises:
            ValueError: On invalid inputs or if both variables appear to be
                continuous (use a Spearman/NPP estimator instead) or if both
                appear to be ordinal (use :class:`PolychoricCorrelation`
                instead).
        """
        x_arr, y_arr = self._prepare(x, y)

        x_is_ord = len(np.unique(x_arr)) < self._n_levels_threshold
        y_is_ord = len(np.unique(y_arr)) < self._n_levels_threshold

        if not x_is_ord and not y_is_ord:
            raise ValueError(
                "Both variables appear continuous (≥ n_levels_threshold unique values). "
                "Use a Spearman or nonparanormal estimator instead."
            )
        if x_is_ord and y_is_ord:
            raise ValueError(
                "Both variables appear ordinal (< n_levels_threshold unique values). Use PolychoricCorrelation instead."
            )

        if x_is_ord:
            cont_arr, ord_arr = y_arr, x_arr
            cont_name, ord_name = "y", "x"
        else:
            cont_arr, ord_arr = x_arr, y_arr
            cont_name, ord_name = "x", "y"

        _validate_continuous(cont_arr, cont_name)
        _validate_ordinal(ord_arr, ord_name)

        self._correlation = self._clip(self._polyserial(cont_arr, ord_arr))
        return self

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _polyserial(self, cont: np.ndarray, disc: np.ndarray) -> float:
        unique_vals, _ = np.unique(disc, return_counts=True)
        threshold_estimate = _thresholds(disc)

        values = np.sort(unique_vals.astype(float))
        interior_thresholds = threshold_estimate[1:-1]
        value_diff = np.diff(values)
        lambda_val: float = float(np.sum(stats.norm.pdf(interior_thresholds) * value_diff))

        if lambda_val == 0:
            raise ValueError(
                "Denominator λ in ad-hoc polyserial estimator is zero. "
                "This can happen when ordinal categories are far apart in "
                "probability space.  Consider recoding the ordinal variable."
            )

        s_disc = float(np.std(disc.astype(float), ddof=1))
        r = _npn_pearson(cont, disc)
        return r * s_disc / lambda_val


# ---------------------------------------------------------------------------
# Module-level public API (kept for backward compatibility)
# ---------------------------------------------------------------------------


def f_hat(x: np.ndarray) -> np.ndarray:
    """Winsorized nonparanormal transformation.

    Implements the estimator from Liu, Lafferty & Wasserman (2009):  each
    observation is mapped to its normal score and the result is scaled to
    unit variance.  Extreme ranks are Winsorized to avoid ±∞.

    Args:
        x: A 1-D numeric array.

    Returns:
        Transformed array of the same length, scaled to unit variance.
    """
    x_arr = _to_array(x)
    if x_arr.size < 2:
        raise ValueError("f_hat requires at least 2 observations.")
    return _f_hat(x_arr)


def npn_pearson(cont: np.ndarray, disc: np.ndarray) -> float:
    """Nonparanormal product-moment correlation.

    Args:
        cont: Continuous numeric array.
        disc: Discrete/ordinal array.

    Returns:
        Correlation coefficient in [−1, 1].
    """
    cont_arr, disc_arr = _to_array(cont), _to_array(disc)
    _validate_pair(cont_arr, disc_arr)
    return _npn_pearson(cont_arr, disc_arr)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's rank correlation coefficient.

    Args:
        x: First numeric array.
        y: Second numeric array (same length as *x*).

    Returns:
        Spearman's ρ in [−1, 1].
    """
    x_arr, y_arr = _to_array(x), _to_array(y)
    _validate_pair(x_arr, y_arr)
    rho, _ = stats.spearmanr(x_arr, y_arr)  # pyright: ignore[reportArgumentType]
    return float(rho)


def adhoc_polyserial(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_cor: float = 0.9999,
    n_levels_threshold: int = 20,
    verbose: bool = False,
) -> float:
    """Convenience dispatcher for mixed-type correlation estimation.

    Selects the appropriate estimator based on the number of unique values:

    * Both continuous → nonparanormal Spearman sin-transformation.
    * Both ordinal → :class:`PolychoricCorrelation`.
    * Mixed → :class:`PolyserialCorrelation`.

    Args:
        x: First array (discrete or continuous).
        y: Second array (discrete or continuous).
        max_cor: Maximum allowed absolute correlation; clips to ±*max_cor*.
        n_levels_threshold: Unique-value count below which a variable is
            treated as ordinal.
        verbose: Emit an :mod:`logging` info message describing the chosen
            estimator.

    Returns:
        Correlation estimate in [−1, 1].
    """
    x_arr, y_arr = _to_array(x), _to_array(y)
    _validate_pair(x_arr, y_arr)

    n_unique_x = len(np.unique(x_arr))
    n_unique_y = len(np.unique(y_arr))
    x_is_discrete = n_unique_x < n_levels_threshold
    y_is_discrete = n_unique_y < n_levels_threshold

    if not x_is_discrete and not y_is_discrete:
        if verbose:
            logger.info("Both variables continuous — using nonparanormal Spearman.")
        rho = spearman(x_arr, y_arr)
        return float(np.clip(2 * np.sin(np.pi / 6 * rho), -max_cor, max_cor))

    if x_is_discrete and y_is_discrete:
        if verbose:
            logger.info("Both variables ordinal — using polychoric correlation.")
        return PolychoricCorrelation(max_cor=max_cor).fit(x_arr, y_arr).correlation

    if verbose:
        logger.info("Mixed pair — using ad-hoc polyserial correlation.")
    return PolyserialCorrelation(max_cor=max_cor, n_levels_threshold=n_levels_threshold).fit(x_arr, y_arr).correlation
