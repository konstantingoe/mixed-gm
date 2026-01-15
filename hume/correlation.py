"""Correlation estimation functions for mixed data.

This module implements correlation estimation methods for mixed continuous
and discrete (ordinal) data, following the latent Gaussian copula approach.

References:
    Liu, Han, Lafferty, John and Wasserman, Larry. (2009).
    The nonparanormal: Semi-parametric estimation of high dimensional
    undirected graphs. Journal of Machine Learning Research 10(80), 2295–2328.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def spearman(x: NDArray[Any], y: NDArray[Any]) -> float:
    """Calculate Spearman's rank correlation coefficient.

    Args:
        x: First numeric array.
        y: Second numeric array.

    Returns:
        Spearman's rho correlation coefficient in [-1, 1].

    Examples:
        >>> import numpy as np
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([5, 6, 7, 8, 7])
        >>> spearman(x, y)  # doctest: +ELLIPSIS
        0.8...
    """
    rank_x = stats.rankdata(x)
    rank_y = stats.rankdata(y)
    rank_mean = (len(rank_x) + 1) / 2

    numerator = np.sum((rank_x - rank_mean) * (rank_y - rank_mean))
    denominator = np.sqrt(np.sum((rank_x - rank_mean) ** 2) * np.sum((rank_y - rank_mean) ** 2))

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def f_hat(x: NDArray[Any]) -> NDArray[np.floating[Any]]:
    """Estimate monotone transformation functions using the Winsorized estimator.

    Implements the nonparanormal transformation from Liu, Lafferty, Wasserman (2009).

    Args:
        x: A numeric array.

    Returns:
        Transformed array of the same length as input, scaled to unit variance.

    References:
        Liu, Han, Lafferty, John and Wasserman, Larry. (2009).
        The nonparanormal: Semi-parametric estimation of high dimensional
        undirected graphs. Journal of Machine Learning Research 10(80), 2295–2328.
    """
    n = len(x)
    npn_thresh = 1 / (4 * (n**0.25) * np.sqrt(np.pi * np.log(n)))

    # Compute ranks and scale to (0, 1)
    ranks = stats.rankdata(x) / n

    # Winsorize to avoid boundary issues
    ranks = np.clip(ranks, npn_thresh, 1 - npn_thresh)

    # Transform to standard normal
    transformed = stats.norm.ppf(ranks)

    # Scale to unit variance
    std = np.std(transformed, ddof=1)
    if std > 0:
        transformed = transformed / std

    return transformed


def npn_pearson(
    cont: NDArray[Any],
    disc: NDArray[Any],
) -> float:
    """Calculate nonparanormal product moment correlation.

    Computes the nonparanormal version of the Pearson correlation between
    a continuous and discrete variable.

    Args:
        cont: Continuous numeric array.
        disc: Discrete/categorical array (can be integer or float).

    Returns:
        Correlation coefficient in [-1, 1].
    """
    f_x = f_hat(cont)
    y = np.asarray(disc, dtype=float)

    return float(np.corrcoef(f_x, y)[0, 1])


def adhoc_polyserial(
    x: NDArray[Any],
    y: NDArray[Any],
    *,
    max_cor: float = 0.9999,
    n_levels_threshold: int = 20,
    verbose: bool = False,
) -> float:
    """Compute adhoc polyserial correlation estimate.

    Returns the nonparanormal correlation estimate of a couple of random variables
    where one is ordinal and the other one is continuous.

    Args:
        x: First array (discrete or continuous).
        y: Second array (discrete or continuous).
        max_cor: Maximum allowed correlation to stay away from boundary.
        n_levels_threshold: Number of unique values below which a variable
            is considered discrete.
        verbose: If True, print extra information.

    Returns:
        Correlation estimate in [-1, 1].
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Determine which variable is discrete (fewer unique values)
    n_unique_x = len(np.unique(x))
    n_unique_y = len(np.unique(y))

    x_is_discrete = n_unique_x < n_levels_threshold
    y_is_discrete = n_unique_y < n_levels_threshold

    # Both continuous: use nonparanormal Spearman
    if not x_is_discrete and not y_is_discrete:
        rho = spearman(x, y)
        return float(2 * np.sin(np.pi / 6 * rho))

    # Both discrete: use polychoric correlation approximation
    if x_is_discrete and y_is_discrete:
        return _polychoric_approx(x, y)

    # Mixed case: one continuous, one discrete
    if x_is_discrete:
        numeric_var = y
        factor_var = x
    else:
        numeric_var = x
        factor_var = y

    if verbose:
        logger.info("Computing adhoc polyserial for mixed continuous-discrete pair")

    n = len(factor_var)

    # Get cumulative marginal proportions
    unique_vals, counts = np.unique(factor_var, return_counts=True)
    cumm_proportions = np.concatenate([[0], np.cumsum(counts / n)])

    # Compute threshold estimates
    threshold_estimate = stats.norm.ppf(cumm_proportions)

    # Compute lambda (adjustment factor)
    values = np.sort(unique_vals.astype(float))
    threshold_diff = np.diff(threshold_estimate[:-1])  # Exclude last inf
    value_diff = np.diff(values)

    # Handle edge cases
    valid_mask = ~np.isinf(threshold_diff)
    if valid_mask.sum() > 0:
        phi_vals = stats.norm.pdf(threshold_estimate[1:-1])
        lambda_val = np.sum(phi_vals[valid_mask] * value_diff[valid_mask])
    else:
        lambda_val = 1.0

    s_disc = np.std(factor_var.astype(float), ddof=1)

    r = npn_pearson(numeric_var, factor_var)

    corr_hat = r * s_disc / lambda_val if lambda_val != 0 else r

    # Clip to valid correlation range
    if np.abs(corr_hat) >= 1:
        corr_hat = np.sign(corr_hat) * max_cor

    return float(corr_hat)


def _polychoric_approx(
    x: NDArray[Any],
    y: NDArray[Any],
) -> float:
    """Approximate polychoric correlation for two discrete variables.

    This is a simplified approximation using the tetrachoric correlation
    approach extended to ordinal variables.

    Args:
        x: First discrete array.
        y: Second discrete array.

    Returns:
        Approximate polychoric correlation in [-1, 1].
    """
    # For ordinal variables, use Spearman as a reasonable approximation
    # This could be replaced with a proper polychoric implementation
    rho = spearman(x.astype(float), y.astype(float))
    return float(2 * np.sin(np.pi / 6 * rho))
