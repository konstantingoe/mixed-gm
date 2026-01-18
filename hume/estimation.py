"""Mixed graph estimation for high-dimensional data.

This module implements the latent Gaussian and latent Gaussian copula modeling
approaches to learning mixed high-dimensional graphs.

Given a (potentially high-dimensional) dataset containing both discrete and
continuous variables, these functions estimate undirected graphs by first
computing a correlation matrix and then applying graphical lasso with
eBIC model selection.

References:
    Foygel, Rina and Drton, Mathias. (2010).
    Extended Bayesian Information Criteria for Gaussian Graphical Models.
    Advances in Neural Information Processing Systems, Volume 23, pp. 604–612.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import linalg, stats
from sklearn.covariance import graphical_lasso

from hume.correlation import adhoc_polyserial

logger = logging.getLogger(__name__)


@dataclass
class MixedGraphResult:
    """Result container for mixed graph estimation.

    Attributes:
        precision_matrix: Estimated precision matrix (partial correlations if partial=True).
        adjacency_matrix: Boolean adjacency matrix encoding the undirected graph.
        correlation_matrix: Sample correlation matrix used in glasso.
        n_edges: Number of edges in the estimated graph.
        max_degree: Largest number of connections across all nodes.
        initial_mat_singular: Whether the initial correlation matrix was singular.
        feature_names: Names of features/nodes (if provided).
    """

    precision_matrix: NDArray[Any]
    adjacency_matrix: NDArray[np.bool_]
    correlation_matrix: NDArray[Any]
    n_edges: int
    max_degree: int
    initial_mat_singular: bool
    feature_names: list[str] | None = None

    def __repr__(self) -> str:
        """Return string representation."""
        n_nodes = self.precision_matrix.shape[0]
        return f"MixedGraphResult(n_nodes={n_nodes}, n_edges={self.n_edges}, max_degree={self.max_degree})"


def edgenumber(precision: NDArray[Any], *, cut: float = 0.0) -> int:
    """Calculate number of edges given a precision matrix.

    Args:
        precision: Square matrix whose entries are (negative) partial correlations.
        cut: Thresholding parameter; entries with absolute value <= cut are
            treated as zero.

    Returns:
        Number of edges in the underlying graph (lower triangular count).
    """
    mask = np.abs(precision) > cut
    return int(np.sum(mask[np.tril_indices(precision.shape[0], k=-1)]))


def _make_positive_definite(
    matrix: NDArray[Any],
    *,
    keep_diag: bool = True,
) -> tuple[NDArray[Any], bool]:
    """Project matrix onto positive definite cone if necessary.

    Args:
        matrix: Input correlation/covariance matrix.
        keep_diag: If True, ensure diagonal remains 1 (for correlation matrices).

    Returns:
        Tuple of (positive definite matrix, was_singular).
    """
    # Check if already positive definite
    try:
        eigvals = linalg.eigvalsh(matrix)
        if np.min(eigvals) > 1e-10:
            return matrix, False
    except linalg.LinAlgError:
        pass

    # Project to nearest positive definite matrix
    was_singular = True

    # Eigenvalue decomposition
    eigvals, eigvecs = linalg.eigh(matrix)

    # Clip negative eigenvalues
    eigvals = np.maximum(eigvals, 1e-10)

    # Reconstruct
    pd_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Ensure symmetry
    pd_matrix = (pd_matrix + pd_matrix.T) / 2

    # Rescale to correlation matrix if needed
    if keep_diag:
        d = np.sqrt(np.diag(pd_matrix))
        pd_matrix = pd_matrix / np.outer(d, d)
        np.fill_diagonal(pd_matrix, 1.0)

    return pd_matrix, was_singular


def _estimate_mixed_correlation_gauss(
    data: pd.DataFrame,
    *,
    n_levels_threshold: int = 20,
    verbose: bool = True,
) -> NDArray[Any]:
    """Estimate correlation matrix for mixed data under Gaussian assumption.

    Uses Pearson correlation for continuous-continuous pairs,
    polyserial correlation for continuous-discrete pairs,
    and polychoric correlation for discrete-discrete pairs.

    Args:
        data: DataFrame with mixed continuous and discrete variables.
        n_levels_threshold: Variables with fewer unique values are treated as discrete.
        verbose: If True, print warnings about data characteristics.

    Returns:
        Estimated correlation matrix.
    """
    d = data.shape[1]
    rho = np.eye(d)

    # Identify discrete columns
    is_discrete = np.array([data.iloc[:, j].nunique() < n_levels_threshold for j in range(d)])

    if not is_discrete.any() and verbose:
        logger.warning(
            "No discrete variables detected (all columns have >= %d unique values). "
            "Consider using standard Gaussian graphical models.",
            n_levels_threshold,
        )

    for i in range(d - 1):
        for j in range(i + 1, d):
            x_i = data.iloc[:, i].to_numpy()
            x_j = data.iloc[:, j].to_numpy()

            if not is_discrete[i] and not is_discrete[j]:
                # Both continuous: Pearson correlation
                rho[i, j] = rho[j, i] = np.corrcoef(x_i, x_j)[0, 1]
            elif is_discrete[i] and is_discrete[j]:
                # Both discrete: polychoric approximation
                rho_s = float(stats.spearmanr(x_i.astype(float), x_j.astype(float))[0])  # pyright: ignore[reportArgumentType]
                rho[i, j] = rho[j, i] = 2 * np.sin(np.pi / 6 * rho_s)
            else:
                # Mixed: polyserial correlation (using adhoc approach)
                rho[i, j] = rho[j, i] = adhoc_polyserial(x_i, x_j, n_levels_threshold=n_levels_threshold)

    return rho


def _estimate_mixed_correlation_nonpara(
    data: pd.DataFrame,
    *,
    n_levels_threshold: int = 20,
    verbose: bool = True,
) -> NDArray[Any]:
    """Estimate correlation matrix for mixed data under nonparanormal assumption.

    Uses the nonparanormal (Gaussian copula) approach:
    - For continuous-continuous: sin(pi/6 * Spearman) transformation
    - For continuous-discrete: adhoc polyserial with npn transformation
    - For discrete-discrete: polychoric correlation

    Args:
        data: DataFrame with mixed continuous and discrete variables.
        n_levels_threshold: Variables with fewer unique values are treated as discrete.
        verbose: If True, print warnings about data characteristics.

    Returns:
        Estimated correlation matrix.
    """
    d = data.shape[1]
    rho = np.eye(d)

    # Identify discrete columns
    is_discrete = np.array([data.iloc[:, j].nunique() < n_levels_threshold for j in range(d)])

    if not is_discrete.any() and verbose:
        logger.warning("No discrete variables detected. This will default to the nonparanormal SKEPTIC.")

    for i in range(d - 1):
        for j in range(i + 1, d):
            x_i = data.iloc[:, i].to_numpy()
            x_j = data.iloc[:, j].to_numpy()

            if not is_discrete[i] and not is_discrete[j]:
                # Both continuous: nonparanormal transformation of Spearman
                rho_spearman = float(stats.spearmanr(x_i, x_j)[0])  # pyright: ignore[reportArgumentType]
                rho[i, j] = rho[j, i] = 2 * np.sin(np.pi / 6 * rho_spearman)
            elif is_discrete[i] and is_discrete[j]:
                # Both discrete: polychoric approximation
                rho_s = float(stats.spearmanr(x_i.astype(float), x_j.astype(float))[0])  # pyright: ignore[reportArgumentType]
                rho[i, j] = rho[j, i] = 2 * np.sin(np.pi / 6 * rho_s)
            else:
                # Mixed: adhoc polyserial with nonparanormal approach
                rho[i, j] = rho[j, i] = adhoc_polyserial(x_i, x_j, n_levels_threshold=n_levels_threshold)

    return rho


def _check_correlation_bounds(rho: NDArray[Any], threshold: float = 0.9) -> None:
    """Check for correlations close to boundary and warn."""
    lower_tri = rho[np.tril_indices(rho.shape[0], k=-1)]
    high_corr_mask = np.abs(lower_tri) > threshold

    if np.any(high_corr_mask):
        warnings.warn(
            f"Found {high_corr_mask.sum()} correlation(s) with |r| > {threshold}. "
            "Precision matrix inverse might be unreliable.",
            stacklevel=3,
        )


def omega_select(
    precision_path: list[NDArray[Any]],
    lambda_path: NDArray[Any],
    n: int,
    s: NDArray[Any],
    *,
    param: float = 0.1,
    partial: bool = True,
) -> NDArray[Any]:
    """Select precision matrix using extended BIC (eBIC).

    For a given glasso path, selects the model with smallest eBIC:
        eBIC_θ = -2 * loglik(Ω|E) + |E| * log(n) + 4 * |E| * θ * log(d)

    Args:
        precision_path: List of precision matrices from glasso path.
        lambda_path: Array of regularization parameters.
        n: Sample size.
        s: Sample covariance/correlation matrix used for estimation.
        param: Theta parameter for additional high-dimensional penalty.
        partial: If True, convert entries to partial correlations.

    Returns:
        Selected precision matrix (optionally as partial correlations).

    References:
        Foygel, Rina and Drton, Mathias. (2010).
        Extended Bayesian Information Criteria for Gaussian Graphical Models.
        Advances in Neural Information Processing Systems, Volume 23, pp. 604–612.
    """
    d = s.shape[0]
    n_lambda = len(lambda_path)
    ebic = np.zeros(n_lambda)

    for ind in range(n_lambda):
        omega = precision_path[ind]

        # Count edges
        n_edges = edgenumber(omega)

        # Compute log-likelihood
        sign, logdet = np.linalg.slogdet(omega)
        if sign <= 0:
            logdet = -np.inf

        loglik = 0.5 * n * (logdet - np.trace(s @ omega))

        # Compute eBIC
        ebic[ind] = -2 * loglik + n_edges * np.log(n) + 4 * n_edges * param * np.log(d)

    # Select best model
    best_idx = np.argmin(ebic)
    omega_hat = precision_path[best_idx].copy()

    if partial:
        # Convert to partial correlations
        omega_standardized = -stats.zscore(omega_hat, axis=0, ddof=1)
        # Actually, we want: -cov2cor(omega)
        d_omega = np.sqrt(np.diag(omega_hat))
        omega_standardized = -omega_hat / np.outer(d_omega, d_omega)
        np.fill_diagonal(omega_standardized, 1.0)
        return omega_standardized

    return omega_hat


def _run_glasso_path(
    cov_matrix: NDArray[Any],
    n_lambdas: int = 50,
) -> tuple[list[NDArray[Any]], NDArray[Any]]:
    """Run graphical lasso over a path of regularization parameters.

    Args:
        cov_matrix: Sample covariance/correlation matrix.
        n_lambdas: Number of regularization parameters to try.

    Returns:
        Tuple of (list of precision matrices, array of lambda values).
    """
    d = cov_matrix.shape[0]

    # Create lambda path (similar to huge package)
    lambda_max = np.max(np.abs(cov_matrix - np.diag(np.diag(cov_matrix))))
    lambda_min = lambda_max * 0.01
    lambda_path = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), n_lambdas))

    precision_path = []

    for alpha in lambda_path:
        try:
            _, precision, _, _ = graphical_lasso(
                cov_matrix,
                alpha=alpha,
                mode="cd",
                max_iter=100,
            )
            precision_path.append(precision)
        except (FloatingPointError, ValueError):
            # If glasso fails, use identity
            precision_path.append(np.eye(d))

    return precision_path, lambda_path


def mixed_graph_gauss(
    data: pd.DataFrame | NDArray[Any],
    *,
    verbose: bool = True,
    n_lambdas: int = 50,
    param: float = 0.1,
    feature_names: list[str] | None = None,
    n_levels_threshold: int = 20,
) -> MixedGraphResult:
    """Estimate mixed graph under Gaussian assumption.

    Estimates precision matrix and adjacency matrix (undirected graph) for
    potentially high-dimensional data containing both discrete and continuous
    variables under the latent Gaussian assumption.

    Args:
        data: Dataset of dimension (n, d). Can be DataFrame or 2D array.
        verbose: If True, print warnings and information.
        n_lambdas: Length of the glasso path.
        param: Value for additional dimensionality penalty in eBIC.
        feature_names: Optional names for features. If None and data is DataFrame,
            uses column names.
        n_levels_threshold: Variables with fewer unique values are treated as discrete.

    Returns:
        MixedGraphResult containing estimated precision matrix, adjacency matrix,
        correlation matrix, and graph statistics.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> # Create mixed data
        >>> n, d = 100, 5
        >>> continuous = np.random.randn(n, 3)
        >>> discrete = np.random.binomial(1, 0.5, (n, 2))
        >>> data = pd.DataFrame(np.hstack([continuous, discrete]))
        >>> result = mixed_graph_gauss(data, verbose=False)
        >>> result.n_edges >= 0
        True
    """
    # Convert to DataFrame
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    n, d = data.shape

    # Get feature names
    if feature_names is None and hasattr(data, "columns"):
        feature_names = list(data.columns.astype(str))

    if verbose:
        n_discrete = sum(data.iloc[:, j].nunique() < n_levels_threshold for j in range(d))
        if n_discrete == 0:
            logger.warning(
                "Warning: no factors in the input data. Checking input and declaring factors for level(x) < %d",
                n_levels_threshold,
            )

    # Estimate correlation matrix
    rho = _estimate_mixed_correlation_gauss(data, n_levels_threshold=n_levels_threshold, verbose=verbose)

    # Check for high correlations
    _check_correlation_bounds(rho)

    # Make positive definite if necessary
    rho_pd, initial_mat_singular = _make_positive_definite(rho, keep_diag=True)

    # Run glasso path
    precision_path, lambda_path = _run_glasso_path(rho_pd, n_lambdas)

    # Select best model using eBIC
    omega_hat = omega_select(precision_path, lambda_path, n, rho_pd, param=param, partial=True)

    # Compute graph statistics
    n_edges = edgenumber(omega_hat)
    max_degree = int(max(np.sum(np.abs(omega_hat) > 0, axis=1) - 1))
    adjacency = np.abs(omega_hat) > 0

    return MixedGraphResult(
        precision_matrix=omega_hat,
        adjacency_matrix=adjacency,
        correlation_matrix=rho_pd,
        n_edges=n_edges,
        max_degree=max_degree,
        initial_mat_singular=initial_mat_singular,
        feature_names=feature_names,
    )


def mixed_graph_nonpara(
    data: pd.DataFrame | NDArray[Any],
    *,
    verbose: bool = True,
    n_lambdas: int = 50,
    param: float = 0.1,
    feature_names: list[str] | None = None,
    n_levels_threshold: int = 20,
) -> MixedGraphResult:
    """Estimate mixed graph under nonparanormal (Gaussian copula) assumption.

    Estimates precision matrix and adjacency matrix (undirected graph) for
    potentially high-dimensional data containing both discrete and continuous
    variables under the latent Gaussian copula assumption.

    This is more general than mixed_graph_gauss and should be preferred unless
    you know the latent variables are truly Gaussian.

    Args:
        data: Dataset of dimension (n, d). Can be DataFrame or 2D array.
        verbose: If True, print warnings and information.
        n_lambdas: Length of the glasso path.
        param: Value for additional dimensionality penalty in eBIC.
        feature_names: Optional names for features. If None and data is DataFrame,
            uses column names.
        n_levels_threshold: Variables with fewer unique values are treated as discrete.

    Returns:
        MixedGraphResult containing estimated precision matrix, adjacency matrix,
        correlation matrix, and graph statistics.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> # Create mixed data
        >>> n, d = 100, 5
        >>> continuous = np.random.randn(n, 3)
        >>> discrete = np.random.binomial(1, 0.5, (n, 2))
        >>> data = pd.DataFrame(np.hstack([continuous, discrete]))
        >>> result = mixed_graph_nonpara(data, verbose=False)
        >>> result.n_edges >= 0
        True
    """
    # Convert to DataFrame
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    n, d = data.shape

    # Get feature names
    if feature_names is None and hasattr(data, "columns"):
        feature_names = list(data.columns.astype(str))

    if verbose:
        n_discrete = sum(data.iloc[:, j].nunique() < n_levels_threshold for j in range(d))
        if n_discrete == 0:
            logger.warning(
                "Warning: no factors in the input data. Checking input and declaring factors for level(x) < %d",
                n_levels_threshold,
            )

    # Estimate correlation matrix using nonparanormal approach
    rho = _estimate_mixed_correlation_nonpara(data, n_levels_threshold=n_levels_threshold, verbose=verbose)

    # Check for high correlations
    _check_correlation_bounds(rho)

    # Make positive definite if necessary
    rho_pd, initial_mat_singular = _make_positive_definite(rho, keep_diag=True)

    # Run glasso path
    precision_path, lambda_path = _run_glasso_path(rho_pd, n_lambdas)

    # Select best model using eBIC
    omega_hat = omega_select(precision_path, lambda_path, n, rho_pd, param=param, partial=True)

    # Compute graph statistics
    n_edges = edgenumber(omega_hat)
    max_degree = int(max(np.sum(np.abs(omega_hat) > 0, axis=1) - 1))
    adjacency = np.abs(omega_hat) > 0

    return MixedGraphResult(
        precision_matrix=omega_hat,
        adjacency_matrix=adjacency,
        correlation_matrix=rho_pd,
        n_edges=n_edges,
        max_degree=max_degree,
        initial_mat_singular=initial_mat_singular,
        feature_names=feature_names,
    )
