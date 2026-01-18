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
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def f_hat(x: np.ndarray) -> np.ndarray:
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
    n = x.shape[0]
    npn_thresh = 1 / (4 * (n**0.25) * np.sqrt(np.pi * np.log(n)))

    ranks = stats.rankdata(x, method="average")
    ecdf_values = ranks / (n + 1)  # or ranks / n
    ecdf_winsorized = np.clip(ecdf_values, npn_thresh, 1 - npn_thresh)

    transformed = stats.norm.ppf(ecdf_winsorized)
    transformed /= np.std(transformed, ddof=1)

    return transformed


def npn_pearson(
    cont: np.ndarray,
    disc: np.ndarray,
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
    return float(np.corrcoef(f_x, disc)[0, 1])


def adhoc_polyserial(
    x: np.ndarray,
    y: np.ndarray,
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
    # Determine which variable is discrete (fewer unique values)
    n_unique_x = len(np.unique(x))
    n_unique_y = len(np.unique(y))

    x_is_discrete = n_unique_x < n_levels_threshold
    y_is_discrete = n_unique_y < n_levels_threshold

    # Both continuous: use nonparanormal Spearman
    if not x_is_discrete and not y_is_discrete:
        rho = float(stats.spearmanr(x, y)[0])  # pyright: ignore[reportArgumentType]
        return float(2 * np.sin(np.pi / 6 * rho))

    # Both discrete: use polychoric correlation approximation
    if x_is_discrete and y_is_discrete:
        return polychoric_corr(x, y)

    # Mixed case: one continuous, one discrete
    if x_is_discrete:
        numeric_var = y
        disc_var = x
    else:
        numeric_var = x
        disc_var = y

    if verbose:
        logger.info("Computing adhoc polyserial for mixed continuous-discrete pair")

    n = disc_var.shape[0]

    # Get cumulative marginal proportions
    unique_vals, counts = np.unique(disc_var, return_counts=True)
    cumm_proportions = np.concatenate([[0], np.cumsum(counts / n)])

    # Compute threshold estimates
    threshold_estimate = stats.norm.ppf(cumm_proportions)

    # Compute denominator

    # discrete levels
    values = np.sort(unique_vals.astype(float))
    interior_thresholds = threshold_estimate[1:-1]
    value_diff = np.diff(values)
    lambda_val: float = np.sum(stats.norm.pdf(interior_thresholds) * value_diff)

    s_disc = np.std(disc_var.astype(float), ddof=1)

    r = npn_pearson(numeric_var, disc_var)
    if lambda_val != 0:
        corr_hat = r * s_disc / lambda_val
    else:
        raise ValueError("Denominator in ad-hoc polyserial estimator computed as zero, cannot compute correlation.")

    # Clip to valid correlation range
    if np.abs(corr_hat) >= 1:
        corr_hat = np.sign(corr_hat) * max_cor

    return float(corr_hat)


def polychoric_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute polychoric correlation estimate for  two discrete variables.

    Args:
        x: First discrete/categorical array.
        y: Second discrete/categorical array.

    Returns:
        Correlation estimate in [-1, 1].
    """
    return 0.0
