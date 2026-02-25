"""Mixed graphical model estimation for high-dimensional data.

Two main classes are provided:

* :class:`SampleCorrelation` — estimates a sample correlation matrix suited
  for mixed continuous/ordinal data, dispatching to the appropriate pairwise
  measure (Pearson, polychoric, polyserial, or nonparanormal variants).

* :class:`MixedGraphicalLasso` — fits a sparse precision matrix and the
  associated undirected graph via graphical lasso with eBIC model selection.
  Mirrors the sklearn ``GraphicalLassoCV`` API but uses :func:`omega_select`
  for regularisation path selection instead of cross-validation.

References:
    Foygel, Rina and Drton, Mathias. (2010).
    Extended Bayesian Information Criteria for Gaussian Graphical Models.
    Advances in Neural Information Processing Systems, Volume 23, pp. 604–612.

    Liu, Han, Lafferty, John and Wasserman, Larry. (2009).
    The nonparanormal: Semi-parametric estimation of high dimensional
    undirected graphs. Journal of Machine Learning Research 10(80), 2295-2328.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import linalg, stats
from sklearn.covariance import graphical_lasso

from humpy.correlation import PolychoricCorrelation, PolyserialCorrelation
from humpy.graphs import UGRAPH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------


def _edgenumber(precision: NDArray[Any], *, cut: float = 0.0) -> int:
    """Number of off-diagonal non-zeros in the lower triangle of *precision*.

    Args:
        precision: Square precision matrix.
        cut: Entries with |value| <= cut are treated as structural zeros.

    Returns:
        Integer edge count.
    """
    mask = np.abs(precision) > cut
    return int(np.sum(mask[np.tril_indices(precision.shape[0], k=-1)]))


def _make_positive_definite(
    matrix: NDArray[Any],
    *,
    keep_diag: bool = True,
) -> tuple[NDArray[Any], bool]:
    """Project *matrix* onto the positive-definite cone when necessary.

    Args:
        matrix: Square symmetric matrix (correlation or covariance).
        keep_diag: Rescale result to a correlation matrix (diagonal = 1).

    Returns:
        Tuple ``(pd_matrix, was_singular)`` where *was_singular* is ``True``
        when the input had non-positive eigenvalues.
    """
    try:
        eigvals = linalg.eigvalsh(matrix)
        if np.min(eigvals) > 1e-10:
            return matrix, False
    except linalg.LinAlgError:
        pass

    was_singular = True
    eigvals, eigvecs = linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, 1e-10)
    pd_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
    pd_matrix = (pd_matrix + pd_matrix.T) / 2

    if keep_diag:
        d = np.sqrt(np.diag(pd_matrix))
        pd_matrix = pd_matrix / np.outer(d, d)
        np.fill_diagonal(pd_matrix, 1.0)

    return pd_matrix, was_singular


def _check_correlation_bounds(rho: NDArray[Any], threshold: float = 0.9) -> None:
    """Warn when any off-diagonal correlation exceeds *threshold* in magnitude.

    Args:
        rho: Correlation matrix.
        threshold: Warning threshold.
    """
    lower_tri = rho[np.tril_indices(rho.shape[0], k=-1)]
    high_mask = np.abs(lower_tri) > threshold
    if np.any(high_mask):
        warnings.warn(
            f"Found {high_mask.sum()} correlation(s) with |r| > {threshold}. "
            "The precision matrix estimate may be unstable.",
            stacklevel=3,
        )


def _glasso_path(
    cov_matrix: NDArray[Any],
    n_lambdas: int = 50,
) -> tuple[list[NDArray[Any]], NDArray[Any]]:
    """Compute the graphical lasso regularisation path.

    Runs :func:`sklearn.covariance.graphical_lasso` for a log-spaced grid of
    ``alpha`` values between ``lambda_max`` and ``0.01 * lambda_max``.

    Args:
        cov_matrix: Sample correlation or covariance matrix.
        n_lambdas: Number of regularisation steps.

    Returns:
        Tuple ``(precision_path, lambda_path)`` where *precision_path* is a
        list of length *n_lambdas* and *lambda_path* is a decreasing array.
    """
    d = cov_matrix.shape[0]
    off_diag = cov_matrix - np.diag(np.diag(cov_matrix))
    lambda_max = np.max(np.abs(off_diag))
    lambda_min = lambda_max * 0.01
    lambda_path = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), n_lambdas))

    precision_path: list[NDArray[Any]] = []
    for alpha in lambda_path:
        try:
            _, precision = graphical_lasso(cov_matrix, alpha=alpha, mode="cd", max_iter=200)
            precision_path.append(precision)
        except (FloatingPointError, ValueError):
            logger.debug("graphical_lasso failed for alpha=%.4f; substituting identity.", alpha)
            precision_path.append(np.eye(d))

    return precision_path, lambda_path


def omega_select(
    precision_path: list[NDArray[Any]],
    lambda_path: NDArray[Any],
    n: int,
    s: NDArray[Any],
    *,
    gamma: float = 0.1,
) -> tuple[NDArray[Any], float, NDArray[Any]]:
    r"""Select the best precision matrix from a glasso path using eBIC.

    The extended BIC criterion is:

    .. math::

        \mathrm{eBIC}(\Omega) =
            -2 \ell(\Omega \mid E)
            + |E| \log n
            + 4 \gamma |E| \log d

    where :math:`|E|` is the number of edges, :math:`n` the sample size,
    :math:`d` the number of variables, and :math:`\gamma \in [0,1]` an
    additional penalty for high-dimensional settings.

    Args:
        precision_path: List of precision matrices along the regularisation path.
        lambda_path: Corresponding array of ``alpha`` values (same length).
        n: Sample size.
        s: Sample correlation/covariance matrix used for estimation.
        gamma: eBIC hyper-parameter.  ``0`` recovers standard BIC.
            Defaults to 0.1.

    Returns:
        Tuple ``(selected_precision, selected_alpha, ebic_scores)`` where
        *selected_precision* is the raw precision matrix for the chosen model,
        *selected_alpha* is the corresponding regularisation value, and
        *ebic_scores* is the full eBIC array along the path.

    References:
        Foygel, Rina and Drton, Mathias. (2010).
        Extended Bayesian Information Criteria for Gaussian Graphical Models.
        Advances in Neural Information Processing Systems, Volume 23, pp. 604-612.
    """
    d = s.shape[0]
    ebic = np.zeros(len(lambda_path))

    for k, omega in enumerate(precision_path):
        n_edges = _edgenumber(omega)
        sign, logdet = np.linalg.slogdet(omega)
        logdet = logdet if sign > 0 else -np.inf
        loglik = 0.5 * n * (logdet - np.trace(s @ omega))
        ebic[k] = -2 * loglik + n_edges * np.log(n) + 4 * gamma * n_edges * np.log(d)

    best = int(np.argmin(ebic))
    return precision_path[best].copy(), float(lambda_path[best]), ebic


def _precision_to_partial(omega: NDArray[Any]) -> NDArray[Any]:
    """Convert a precision matrix to partial correlations.

    Standardises off-diagonal entries as
    ``-omega[i,j] / sqrt(omega[i,i] * omega[j,j])``.

    Args:
        omega: Positive definite precision matrix.

    Returns:
        Partial correlation matrix with unit diagonal.
    """
    d_inv = np.sqrt(np.diag(omega))
    pcor: NDArray[Any] = np.asarray(-omega / np.outer(d_inv, d_inv))
    np.fill_diagonal(pcor, 1.0)
    return pcor


# ---------------------------------------------------------------------------
# SampleCorrelation
# ---------------------------------------------------------------------------


class SampleCorrelation:
    r"""Estimate a sample correlation matrix for mixed continuous/ordinal data.

    Pairwise correlations are estimated as follows:

    * **continuous - continuous**: Spearman sin-transform
      :math:`\hat{\sigma} = 2 \sin(\pi/6 \cdot \hat{\rho}_S)`.
    * **continuous - ordinal**: ad-hoc polyserial correlation via
      :class:`~humpy.correlation.PolyserialCorrelation`.
    * **ordinal - ordinal**: maximum-likelihood polychoric correlation via
      :class:`~humpy.correlation.PolychoricCorrelation`.

    After calling :meth:`fit`, the estimated matrix is available as
    :attr:`correlation_matrix_`.

    Args:
        n_levels_threshold: Columns with strictly fewer unique values are
            treated as ordinal.  Defaults to 20.

    Example::

        sc = SampleCorrelation().fit(X)
        print(sc.correlation_matrix_)
    """

    def __init__(
        self,
        n_levels_threshold: int = 20,
    ) -> None:
        """Construct a SampleCorrelation estimator.

        Args:
            n_levels_threshold: Unique-value threshold for ordinal detection.
                Defaults to 20.
        """
        self.n_levels_threshold = n_levels_threshold

        # set after fit
        self.correlation_matrix_: pd.DataFrame | None = None
        self.feature_names_: list[str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, x: pd.DataFrame | NDArray[Any]) -> SampleCorrelation:
        """Estimate the correlation matrix from *x*.

        Args:
            x: Data matrix of shape ``(n_samples, n_features)``.
                A :class:`pandas.DataFrame` is preferred so that column names
                are preserved; a 2-D :class:`numpy.ndarray` is also accepted.

        Returns:
            *self* -- enables method chaining.

        Raises:
            ValueError: If *x* has fewer than 2 columns.
        """
        df = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)
        if df.shape[1] < 2:
            raise ValueError("X must have at least 2 columns.")

        self.feature_names_ = list(df.columns.astype(str))

        rho = self._estimate(df)

        _check_correlation_bounds(rho)
        self.correlation_matrix_ = pd.DataFrame(rho, index=self.feature_names_, columns=self.feature_names_)
        return self

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_discrete(self, df: pd.DataFrame) -> NDArray[np.bool_]:
        return np.array([df.iloc[:, j].nunique() < self.n_levels_threshold for j in range(df.shape[1])])

    def _estimate(self, df: pd.DataFrame) -> NDArray[Any]:
        r"""Estimate pairwise correlations under the latent Gaussian copula.

        Rules per pair type:

        * **cont - cont**: :math:`2 \sin(\pi/6 \cdot \hat{\rho}_S)` where
            :math:`\hat{\rho}_S` is Spearman's rank correlation.
        * **cont - ord**: ad-hoc polyserial via :class:`PolyserialCorrelation`.
        * **ord - ord**: MLE polychoric via :class:`PolychoricCorrelation`.
        """
        d = df.shape[1]
        rho = np.eye(d)
        disc = self._is_discrete(df)

        for i in range(d - 1):
            for j in range(i + 1, d):
                xi = df.iloc[:, i].to_numpy(dtype=float)
                xj = df.iloc[:, j].to_numpy(dtype=float)

                if not disc[i] and not disc[j]:
                    rho_s = float(stats.spearmanr(xi, xj)[0])  # pyright: ignore[reportArgumentType]
                    r = 2.0 * np.sin(np.pi / 6.0 * rho_s)
                elif disc[i] and disc[j]:
                    r = PolychoricCorrelation().fit(xi, xj).correlation
                else:
                    r = PolyserialCorrelation(n_levels_threshold=self.n_levels_threshold).fit(xi, xj).correlation

                rho[i, j] = rho[j, i] = r

        return rho


# ---------------------------------------------------------------------------
# MixedGraphicalLasso
# ---------------------------------------------------------------------------


class MixedGraphicalLasso:
    r"""Sparse Gaussian graphical model for mixed continuous/ordinal data.

    Estimates a precision matrix and the associated :class:`~humpy.graphs.UGRAPH`
    by:

    1. Computing the sample correlation matrix via :class:`SampleCorrelation`.
    2. Projecting to the positive-definite cone if necessary.
    3. Computing the full graphical lasso regularisation path.
    4. Selecting the model with the smallest eBIC via :func:`omega_select`.
    5. Re-fitting graphical lasso at the selected alpha for a clean final solution.

    The overall design mirrors ``sklearn.covariance.GraphicalLassoCV`` but
    replaces cross-validation with eBIC-based selection.

    Args:
        n_lambdas: Number of regularisation steps in the path.  Defaults to 50.
        ebic_gamma: eBIC hyper-parameter :math:`\gamma`.  ``0`` gives standard
            BIC; larger values impose a stronger dimensionality penalty.
            Defaults to 0.1.
        n_levels_threshold: Unique-value threshold for declaring a column
            ordinal.  Defaults to 20.

    Attributes set after :meth:`fit`:
        precision_matrix_ (pd.DataFrame): Partial correlation matrix derived
            from the estimated precision (diagonal = 1).
        correlation_matrix_ (pd.DataFrame): Sample correlation matrix used as
            input to graphical lasso.
        graph_ (UGRAPH): Undirected graph whose edges correspond to non-zero
            off-diagonal entries in *precision_matrix_*.
        alpha_ (float): Selected regularisation parameter.
        ebic_scores_ (np.ndarray): eBIC values for each point on the path.
        singular_ (bool): Whether positive-definite projection was needed.
        feature_names_ (list[str]): Variable names inferred from the input.

    Example::

        mgl = MixedGraphicalLasso(ebic_gamma=0.1)
        mgl.fit(X)
        print(mgl.graph_.edges)
        print(mgl.precision_matrix_)
    """

    def __init__(
        self,
        n_lambdas: int = 50,
        ebic_gamma: float = 0.1,
        n_levels_threshold: int = 20,
    ) -> None:
        """Construct a MixedGraphicalLasso estimator.

        Args:
            n_lambdas: Length of the regularisation path.  Defaults to 50.
            ebic_gamma: eBIC dimensionality penalty weight in [0, 1].
                Defaults to 0.1.
            n_levels_threshold: Ordinal detection threshold.  Defaults to 20.

        Raises:
            ValueError: If *ebic_gamma* is not in [0, 1].
        """
        if not (0.0 <= ebic_gamma <= 1.0):
            raise ValueError(f"`ebic_gamma` must be in [0, 1], got {ebic_gamma}.")

        self.n_lambdas = n_lambdas
        self.ebic_gamma = ebic_gamma
        self.n_levels_threshold = n_levels_threshold

        # set after fit
        self.precision_matrix_: pd.DataFrame | None = None
        self.correlation_matrix_: pd.DataFrame | None = None
        self.graph_: UGRAPH | None = None
        self.alpha_: float | None = None
        self.ebic_scores_: NDArray[Any] | None = None
        self.singular_: bool | None = None
        self.feature_names_: list[str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, x: pd.DataFrame | NDArray[Any]) -> MixedGraphicalLasso:
        """Fit the model to *x*.

        Args:
            x: Data matrix of shape ``(n_samples, n_features)``.
                A :class:`pandas.DataFrame` with named columns is preferred.

        Returns:
            *self* -- enables method chaining.
        """
        df = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)
        n, _ = df.shape

        # ----------------------------------------------------------------
        # Step 1 -- sample correlation
        # ----------------------------------------------------------------
        sc = SampleCorrelation(
            n_levels_threshold=self.n_levels_threshold,
        ).fit(df)

        self.feature_names_ = sc.feature_names_
        rho_raw: NDArray[Any] = sc.correlation_matrix_.to_numpy()  # type: ignore[union-attr]

        # ----------------------------------------------------------------
        # Step 2 -- ensure positive definiteness
        # ----------------------------------------------------------------
        rho, singular = _make_positive_definite(rho_raw, keep_diag=True)
        self.singular_ = singular
        self.correlation_matrix_ = pd.DataFrame(rho, index=self.feature_names_, columns=self.feature_names_)

        # ----------------------------------------------------------------
        # Step 3 -- regularisation path
        # ----------------------------------------------------------------
        precision_path, lambda_path = _glasso_path(rho, n_lambdas=self.n_lambdas)

        # ----------------------------------------------------------------
        # Step 4 -- eBIC model selection
        # ----------------------------------------------------------------
        _, selected_alpha, ebic_scores = omega_select(
            precision_path,
            lambda_path,
            n=n,
            s=rho,
            gamma=self.ebic_gamma,
        )
        self.alpha_ = selected_alpha
        self.ebic_scores_ = ebic_scores

        # ----------------------------------------------------------------
        # Step 5 -- final fit at selected alpha
        # ----------------------------------------------------------------
        try:
            _, omega_final = graphical_lasso(rho, alpha=selected_alpha, mode="cd", max_iter=200)
        except (FloatingPointError, ValueError):
            logger.warning(
                "Final graphical_lasso fit failed for alpha=%.4f; falling back to path estimate.",
                selected_alpha,
            )
            best_idx = int(np.argmin(ebic_scores))
            omega_final = precision_path[best_idx]

        omega_pcor = _precision_to_partial(omega_final)

        # ----------------------------------------------------------------
        # Store results
        # ----------------------------------------------------------------
        names = self.feature_names_
        assert names is not None  # guaranteed by SampleCorrelation.fit
        self.precision_matrix_ = pd.DataFrame(omega_pcor, index=names, columns=names)
        self.graph_ = self._build_ugraph(omega_pcor, names)

        return self

    @property
    def n_edges_(self) -> int:
        """Number of edges in the estimated graph.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if self.graph_ is None:
            raise RuntimeError("Call .fit() before accessing n_edges_.")
        return self.graph_.num_edges

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_ugraph(pcor: NDArray[Any], names: list[str]) -> UGRAPH:
        """Construct a UGRAPH from a partial correlation matrix.

        An edge ``(i, j)`` is included whenever ``pcor[i, j] != 0``.

        Args:
            pcor: Partial correlation matrix (diagonal = 1).
            names: Variable names corresponding to rows/columns.

        Returns:
            UGRAPH with one node per variable and one undirected edge per
            non-zero off-diagonal entry.
        """
        d = pcor.shape[0]
        edges: list[tuple[str, str]] = [
            (names[i], names[j]) for i in range(d - 1) for j in range(i + 1, d) if pcor[i, j] != 0.0
        ]
        return UGRAPH(nodes=names, edges=edges)


# ---------------------------------------------------------------------------
# Backward-compatible API (wraps the new class-based interface)
# ---------------------------------------------------------------------------


from dataclasses import dataclass  # noqa: E402


@dataclass
class MixedGraphResult:
    """Legacy result container kept for backward compatibility.

    New code should use :class:`MixedGraphicalLasso` directly and access
    results via its attributes.

    Attributes:
        precision_matrix: Estimated partial correlation matrix as a numpy array.
        adjacency_matrix: Boolean adjacency matrix (True = edge present).
        correlation_matrix: Sample correlation matrix fed to graphical lasso.
        n_edges: Number of edges in the estimated graph.
        max_degree: Largest node degree.
        initial_mat_singular: Whether the correlation matrix needed PD projection.
        feature_names: Variable names (None if not provided).
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

    Thin wrapper around :func:`_edgenumber` kept for backward compatibility.

    Args:
        precision: Square precision or partial-correlation matrix.
        cut: Entries with |value| <= *cut* are treated as zero.

    Returns:
        Number of edges (lower-triangular non-zeros).
    """
    return _edgenumber(precision, cut=cut)


def _legacy_fit(
    data: pd.DataFrame | NDArray[Any],
    *,
    verbose: bool = True,
    n_lambdas: int = 50,
    param: float = 0.1,
    feature_names: list[str] | None = None,
    n_levels_threshold: int = 20,
) -> MixedGraphResult:
    """Shared implementation for the legacy function wrappers."""
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    if feature_names is not None:
        df = df.copy()
        df.columns = pd.Index(feature_names)

    if verbose:
        d = df.shape[1]
        n_discrete = sum(df.iloc[:, j].nunique() < n_levels_threshold for j in range(d))
        if n_discrete == 0:
            logger.warning(
                "No discrete variables detected (all columns have >= %d unique values).",
                n_levels_threshold,
            )

    mgl = MixedGraphicalLasso(
        n_lambdas=n_lambdas,
        ebic_gamma=param,
        n_levels_threshold=n_levels_threshold,
    ).fit(df)

    assert mgl.precision_matrix_ is not None
    assert mgl.correlation_matrix_ is not None
    assert mgl.singular_ is not None

    omega_arr: NDArray[Any] = mgl.precision_matrix_.to_numpy()
    rho_arr: NDArray[Any] = mgl.correlation_matrix_.to_numpy()
    adjacency: NDArray[np.bool_] = np.abs(omega_arr) > 0
    np.fill_diagonal(adjacency, False)
    adjacency = adjacency | adjacency.T

    n_edges = _edgenumber(omega_arr)
    degrees = np.sum(adjacency, axis=1)
    max_degree = int(np.max(degrees)) if degrees.size > 0 else 0

    return MixedGraphResult(
        precision_matrix=omega_arr,
        adjacency_matrix=adjacency,
        correlation_matrix=rho_arr,
        n_edges=n_edges,
        max_degree=max_degree,
        initial_mat_singular=mgl.singular_,
        feature_names=mgl.feature_names_,
    )


def mixed_graph_gauss(
    data: pd.DataFrame | NDArray[Any],
    *,
    verbose: bool = True,
    n_lambdas: int = 50,
    param: float = 0.1,
    feature_names: list[str] | None = None,
    n_levels_threshold: int = 20,
) -> MixedGraphResult:
    """Estimate a mixed graphical model (backward-compatible wrapper).

    .. deprecated::
        Prefer :class:`MixedGraphicalLasso` directly.

    Args:
        data: Data matrix ``(n_samples, n_features)``.
        verbose: Emit logging warnings.
        n_lambdas: Length of the regularisation path.
        param: eBIC dimensionality penalty weight (gamma).
        feature_names: Override column names.
        n_levels_threshold: Ordinal detection threshold.

    Returns:
        :class:`MixedGraphResult`
    """
    return _legacy_fit(
        data,
        verbose=verbose,
        n_lambdas=n_lambdas,
        param=param,
        feature_names=feature_names,
        n_levels_threshold=n_levels_threshold,
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
    """Estimate a mixed graphical model (backward-compatible wrapper).

    .. deprecated::
        Prefer :class:`MixedGraphicalLasso` directly.

    Args:
        data: Data matrix ``(n_samples, n_features)``.
        verbose: Emit logging warnings.
        n_lambdas: Length of the regularisation path.
        param: eBIC dimensionality penalty weight (gamma).
        feature_names: Override column names.
        n_levels_threshold: Ordinal detection threshold.

    Returns:
        :class:`MixedGraphResult`
    """
    return _legacy_fit(
        data,
        verbose=verbose,
        n_lambdas=n_lambdas,
        param=param,
        feature_names=feature_names,
        n_levels_threshold=n_levels_threshold,
    )
