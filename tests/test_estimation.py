"""Tests for hume estimation module.

Covers:
    - SampleCorrelation: pairwise dispatch, symmetry, unit diagonal
    - omega_select: return types, eBIC monotonicity
    - MixedGraphicalLasso: shapes, graph consistency, sparsity control
    - edgenumber: edge-counting utility
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.covariance import graphical_lasso

from hume.estimation import (
    MixedGraphicalLasso,
    SampleCorrelation,
    edgenumber,
    omega_select,
)
from hume.graphs import UGRAPH
from hume.correlation import PolychoricCorrelation, PolyserialCorrelation


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=2026)


def _make_continuous(rng: np.random.Generator, n: int = 200, d: int = 4) -> pd.DataFrame:
    """Purely continuous data drawn i.i.d. from N(0,1)."""
    return pd.DataFrame(rng.standard_normal((n, d)), columns=[f"x{i}" for i in range(d)])


def _make_mixed(rng: np.random.Generator, n: int = 200) -> pd.DataFrame:
    """Three continuous + two ordinal (0-4) variables."""
    cont = rng.standard_normal((n, 3))
    ordinal = rng.integers(0, 5, size=(n, 2))
    data = np.hstack([cont, ordinal])
    return pd.DataFrame(data, columns=["c0", "c1", "c2", "o0", "o1"])


def _make_two_binary(rng: np.random.Generator, rho: float = 0.7, n: int = 400) -> pd.DataFrame:
    """Two correlated binary variables from latent bivariate normal."""
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n)
    return pd.DataFrame({"b0": (z[:, 0] > 0).astype(float), "b1": (z[:, 1] > 0).astype(float)})


def _make_cont_ord(rng: np.random.Generator, rho: float = 0.7, n: int = 400) -> pd.DataFrame:
    """One continuous + one 3-level ordinal with latent correlation *rho*."""
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n)
    cont = z[:, 0]
    ordinal = (z[:, 1] > -0.5).astype(float) + (z[:, 1] > 0.5).astype(float)
    return pd.DataFrame({"cont": cont, "ord": ordinal})


# ---------------------------------------------------------------------------
# edgenumber
# ---------------------------------------------------------------------------


class TestEdgenumber:
    def test_identity_has_no_edges(self) -> None:
        assert edgenumber(np.eye(5)) == 0

    def test_complete_graph(self) -> None:
        # fully off-diagonal non-zero 5x5: 5*4/2 = 10 edges
        assert edgenumber(np.ones((5, 5))) == 10

    def test_two_edges(self) -> None:
        m = np.eye(4)
        m[0, 1] = m[1, 0] = 0.5
        m[2, 3] = m[3, 2] = 0.3
        assert edgenumber(m) == 2

    def test_cut_threshold(self) -> None:
        m = np.eye(3)
        m[0, 1] = m[1, 0] = 0.3
        m[0, 2] = m[2, 0] = 0.6
        assert edgenumber(m, cut=0.4) == 1


# ---------------------------------------------------------------------------
# SampleCorrelation
# ---------------------------------------------------------------------------


class TestSampleCorrelation:
    """Tests for SampleCorrelation.fit()."""

    # ------ output structure -----------------------------------------------

    def test_returns_self_for_chaining(self, rng: np.random.Generator) -> None:
        sc = SampleCorrelation()
        df = _make_continuous(rng)
        assert sc.fit(df) is sc

    def test_correlation_matrix_is_dataframe(self, rng: np.random.Generator) -> None:
        df = _make_continuous(rng)
        sc = SampleCorrelation().fit(df)
        assert isinstance(sc.correlation_matrix_, pd.DataFrame)

    def test_shape(self, rng: np.random.Generator) -> None:
        df = _make_continuous(rng, d=5)
        sc = SampleCorrelation().fit(df)
        assert sc.correlation_matrix_.shape == (5, 5)

    def test_unit_diagonal(self, rng: np.random.Generator) -> None:
        df = _make_mixed(rng)
        sc = SampleCorrelation().fit(df)
        rho = sc.correlation_matrix_.to_numpy()
        assert np.allclose(np.diag(rho), 1.0)

    def test_symmetry(self, rng: np.random.Generator) -> None:
        df = _make_mixed(rng)
        sc = SampleCorrelation().fit(df)
        rho = sc.correlation_matrix_.to_numpy()
        assert np.allclose(rho, rho.T)

    def test_entries_in_minus_one_plus_one(self, rng: np.random.Generator) -> None:
        df = _make_mixed(rng)
        sc = SampleCorrelation().fit(df)
        rho = sc.correlation_matrix_.to_numpy()
        assert np.all(rho >= -1.0 - 1e-9) and np.all(rho <= 1.0 + 1e-9)

    def test_feature_names_preserved(self, rng: np.random.Generator) -> None:
        df = _make_mixed(rng)
        sc = SampleCorrelation().fit(df)
        assert sc.feature_names_ == list(df.columns)
        assert list(sc.correlation_matrix_.columns) == list(df.columns)

    def test_accepts_numpy_array(self, rng: np.random.Generator) -> None:
        arr = rng.standard_normal((100, 3))
        sc = SampleCorrelation().fit(arr)
        assert sc.correlation_matrix_.shape == (3, 3)

    def test_too_few_columns_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            SampleCorrelation().fit(pd.DataFrame({"a": [1, 2, 3]}))

    # ------ pairwise dispatch ----------------------------------------------

    def test_cont_cont_sin_transform(self, rng: np.random.Generator) -> None:
        """Continuous-continuous uses 2*sin(pi/6 * rho_S); result matches manual calc."""
        df = _make_continuous(rng, n=500, d=2)
        sc = SampleCorrelation().fit(df)
        rho_est = sc.correlation_matrix_.iloc[0, 1]

        rho_s = float(stats.spearmanr(df.iloc[:, 0], df.iloc[:, 1])[0])
        expected = 2.0 * np.sin(np.pi / 6.0 * rho_s)
        assert rho_est == pytest.approx(expected, abs=1e-10)

    def test_ord_ord_uses_polychoric(self, rng: np.random.Generator) -> None:
        """Ordinal-ordinal should use the MLE polychoric, not the sin-transform."""
        df = _make_two_binary(rng, rho=0.7, n=600)
        sc = SampleCorrelation(n_levels_threshold=5).fit(df)
        rho_est = sc.correlation_matrix_.iloc[0, 1]

        expected = PolychoricCorrelation().fit(df["b0"].to_numpy(), df["b1"].to_numpy()).correlation
        assert rho_est == pytest.approx(expected, abs=1e-10)

    def test_cont_ord_uses_polyserial(self, rng: np.random.Generator) -> None:
        """Mixed pair should use PolyserialCorrelation."""
        df = _make_cont_ord(rng, rho=0.6, n=500)
        sc = SampleCorrelation(n_levels_threshold=5).fit(df)
        rho_est = sc.correlation_matrix_.iloc[0, 1]

        expected = (
            PolyserialCorrelation(n_levels_threshold=5).fit(df["cont"].to_numpy(), df["ord"].to_numpy()).correlation
        )
        assert rho_est == pytest.approx(expected, abs=1e-10)

    def test_cont_cont_positive_correlation_recovered(self, rng: np.random.Generator) -> None:
        """Strong positive latent correlation should yield a positive estimate."""
        z = rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], size=600)
        df = pd.DataFrame(z, columns=["a", "b"])
        sc = SampleCorrelation().fit(df)
        assert sc.correlation_matrix_.loc["a", "b"] > 0.4

    def test_n_levels_threshold_controls_discreteness(self, rng: np.random.Generator) -> None:
        """A variable with 10 unique values: continuous at threshold=5, ordinal at threshold=20."""
        arr = rng.integers(0, 10, size=(300, 2)).astype(float)
        df = pd.DataFrame(arr, columns=["a", "b"])

        sc_low = SampleCorrelation(n_levels_threshold=5).fit(df)
        sc_high = SampleCorrelation(n_levels_threshold=20).fit(df)

        rho_s = float(stats.spearmanr(df["a"], df["b"])[0])
        expected_sin = 2.0 * np.sin(np.pi / 6.0 * rho_s)
        assert sc_low.correlation_matrix_.iloc[0, 1] == pytest.approx(expected_sin, abs=1e-10)

        expected_poly = PolychoricCorrelation().fit(df["a"].to_numpy(), df["b"].to_numpy()).correlation
        assert sc_high.correlation_matrix_.iloc[0, 1] == pytest.approx(expected_poly, abs=1e-10)


# ---------------------------------------------------------------------------
# omega_select
# ---------------------------------------------------------------------------


class TestOmegaSelect:
    def _path(self, d: int = 5, n_lambdas: int = 10) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        s = np.eye(d) * 0.8 + rng.standard_normal((d, d)) * 0.05
        s = (s + s.T) / 2
        np.fill_diagonal(s, 1.0)
        lambdas = np.linspace(0.5, 0.01, n_lambdas)
        path = [np.eye(d) * (1.0 + lam) for lam in lambdas]
        return path, lambdas, s

    def test_returns_three_tuple(self) -> None:
        path, lambdas, s = self._path()
        result = omega_select(path, lambdas, n=100, s=s)
        assert len(result) == 3

    def test_selected_precision_is_ndarray(self) -> None:
        path, lambdas, s = self._path()
        omega, _, _ = omega_select(path, lambdas, n=100, s=s)
        assert isinstance(omega, np.ndarray)

    def test_selected_alpha_is_in_path(self) -> None:
        path, lambdas, s = self._path()
        _, alpha, _ = omega_select(path, lambdas, n=100, s=s)
        assert alpha in lambdas

    def test_ebic_scores_length(self) -> None:
        n_lambdas = 10
        path, lambdas, s = self._path(n_lambdas=n_lambdas)
        _, _, ebic = omega_select(path, lambdas, n=100, s=s)
        assert len(ebic) == n_lambdas

    def test_higher_gamma_selects_sparser_model(self) -> None:
        """Higher gamma penalises edges more strongly -> no denser than lower gamma."""

        rng = np.random.default_rng(7)
        d, n = 8, 300
        s = np.corrcoef(rng.standard_normal((n, d)).T)

        lmax = np.max(np.abs(s - np.diag(np.diag(s))))
        lambdas = np.exp(np.linspace(np.log(lmax), np.log(lmax * 0.01), 30))
        path = []
        for lam in lambdas:
            try:
                _, omega, _, _ = graphical_lasso(s, alpha=lam)
                path.append(omega)
            except Exception:
                path.append(np.eye(d))

        omega_low, _, _ = omega_select(path, lambdas, n=n, s=s, gamma=0.0)
        omega_high, _, _ = omega_select(path, lambdas, n=n, s=s, gamma=1.0)
        assert edgenumber(omega_high) <= edgenumber(omega_low)


# ---------------------------------------------------------------------------
# MixedGraphicalLasso
# ---------------------------------------------------------------------------


class TestMixedGraphicalLasso:
    """Tests for MixedGraphicalLasso.fit()."""

    # ------ output structure -----------------------------------------------

    def test_returns_self_for_chaining(self, rng: np.random.Generator) -> None:
        mgl = MixedGraphicalLasso()
        assert mgl.fit(_make_continuous(rng)) is mgl

    def test_precision_matrix_is_dataframe(self, rng: np.random.Generator) -> None:
        mgl = MixedGraphicalLasso().fit(_make_continuous(rng, d=4))
        assert isinstance(mgl.precision_matrix_, pd.DataFrame)

    def test_precision_shape(self, rng: np.random.Generator) -> None:
        d = 4
        mgl = MixedGraphicalLasso().fit(_make_continuous(rng, d=d))
        assert mgl.precision_matrix_.shape == (d, d)

    def test_precision_unit_diagonal(self, rng: np.random.Generator) -> None:
        mgl = MixedGraphicalLasso().fit(_make_mixed(rng))
        omega = mgl.precision_matrix_.to_numpy()
        assert np.allclose(np.diag(omega), 1.0)

    def test_precision_is_symmetric(self, rng: np.random.Generator) -> None:
        mgl = MixedGraphicalLasso().fit(_make_mixed(rng))
        omega = mgl.precision_matrix_.to_numpy()
        assert np.allclose(omega, omega.T, atol=1e-10)

    def test_correlation_matrix_shape(self, rng: np.random.Generator) -> None:
        mgl = MixedGraphicalLasso().fit(_make_mixed(rng))
        assert mgl.correlation_matrix_.shape == (5, 5)

    def test_column_and_index_names(self, rng: np.random.Generator) -> None:
        df = _make_mixed(rng)
        mgl = MixedGraphicalLasso().fit(df)
        assert list(mgl.precision_matrix_.columns) == list(df.columns)
        assert list(mgl.precision_matrix_.index) == list(df.columns)

    def test_feature_names(self, rng: np.random.Generator) -> None:
        df = _make_mixed(rng)
        mgl = MixedGraphicalLasso().fit(df)
        assert mgl.feature_names_ == list(df.columns)

    def test_alpha_is_positive_float(self, rng: np.random.Generator) -> None:
        mgl = MixedGraphicalLasso().fit(_make_continuous(rng, d=4))
        assert isinstance(mgl.alpha_, float)
        assert mgl.alpha_ > 0

    def test_ebic_scores_length(self, rng: np.random.Generator) -> None:
        n_lambdas = 20
        mgl = MixedGraphicalLasso(n_lambdas=n_lambdas).fit(_make_continuous(rng, d=4))
        assert len(mgl.ebic_scores_) == n_lambdas

    def test_singular_flag_is_bool(self, rng: np.random.Generator) -> None:
        mgl = MixedGraphicalLasso().fit(_make_continuous(rng, d=4))
        assert isinstance(mgl.singular_, bool)

    # ------ graph_ attribute -----------------------------------------------

    def test_graph_is_ugraph(self, rng: np.random.Generator) -> None:
        mgl = MixedGraphicalLasso().fit(_make_continuous(rng, d=4))
        assert isinstance(mgl.graph_, UGRAPH)

    def test_graph_nodes_match_features(self, rng: np.random.Generator) -> None:
        df = _make_mixed(rng)
        mgl = MixedGraphicalLasso().fit(df)
        assert set(mgl.graph_.nodes) == set(df.columns)

    def test_graph_edges_consistent_with_precision(self, rng: np.random.Generator) -> None:
        """Every graph edge must correspond to a non-zero off-diagonal precision entry."""
        df = _make_continuous(rng, d=5)
        mgl = MixedGraphicalLasso().fit(df)
        omega = mgl.precision_matrix_.to_numpy()
        names = mgl.feature_names_
        assert names is not None
        name_to_idx = {n: i for i, n in enumerate(names)}
        for u, v in mgl.graph_.edges:
            i, j = name_to_idx[u], name_to_idx[v]
            assert omega[i, j] != 0.0, f"Edge ({u},{v}) exists but precision entry is zero"

    def test_n_edges_property_matches_graph(self, rng: np.random.Generator) -> None:
        mgl = MixedGraphicalLasso().fit(_make_continuous(rng, d=5))
        assert mgl.n_edges_ == mgl.graph_.num_edges

    def test_n_edges_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            _ = MixedGraphicalLasso().n_edges_

    # ------ sparsity control -----------------------------------------------

    def test_higher_gamma_yields_sparser_graph(self, rng: np.random.Generator) -> None:
        """Higher ebic_gamma -> at most as many edges as lower ebic_gamma."""
        df = _make_continuous(rng, n=300, d=6)
        mgl_low = MixedGraphicalLasso(ebic_gamma=0.0, n_lambdas=30).fit(df)
        mgl_high = MixedGraphicalLasso(ebic_gamma=1.0, n_lambdas=30).fit(df)
        assert mgl_high.n_edges_ <= mgl_low.n_edges_

    def test_invalid_gamma_raises(self) -> None:
        with pytest.raises(ValueError, match="ebic_gamma"):
            MixedGraphicalLasso(ebic_gamma=1.5)

    # ------ input handling -------------------------------------------------

    def test_accepts_numpy_array(self, rng: np.random.Generator) -> None:
        arr = rng.standard_normal((150, 4))
        mgl = MixedGraphicalLasso().fit(arr)
        assert mgl.precision_matrix_.shape == (4, 4)

    def test_mixed_data(self, rng: np.random.Generator) -> None:
        df = _make_mixed(rng)
        mgl = MixedGraphicalLasso().fit(df)
        assert mgl.precision_matrix_.shape == (5, 5)
        assert isinstance(mgl.graph_, UGRAPH)


# ---------------------------------------------------------------------------
# Integration: structure recovery
# ---------------------------------------------------------------------------


class TestStructureRecovery:
    """Verify that a strongly linked pair gets detected as an edge."""

    def test_correlated_continuous_pair_detected(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        x0 = rng.standard_normal(n)
        x1 = 0.85 * x0 + 0.53 * rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        x3 = rng.standard_normal(n)
        df = pd.DataFrame({"x0": x0, "x1": x1, "x2": x2, "x3": x3})

        mgl = MixedGraphicalLasso(ebic_gamma=0.1, n_lambdas=40).fit(df)
        edges = mgl.graph_.edges
        assert ("x0", "x1") in edges or ("x1", "x0") in edges

    def test_correlated_binary_pair_detected(self) -> None:
        rng = np.random.default_rng(7)
        n = 600
        z = rng.multivariate_normal([0, 0], [[1, 0.85], [0.85, 1]], size=n)
        b0 = (z[:, 0] > 0).astype(float)
        b1 = (z[:, 1] > 0).astype(float)
        noise = rng.standard_normal((n, 2))
        df = pd.DataFrame({"b0": b0, "b1": b1, "n0": noise[:, 0], "n1": noise[:, 1]})

        mgl = MixedGraphicalLasso(ebic_gamma=0.1, n_lambdas=40).fit(df)
        edges = mgl.graph_.edges
        assert ("b0", "b1") in edges or ("b1", "b0") in edges
