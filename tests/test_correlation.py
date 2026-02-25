"""Tests for humpy correlation functions."""

import numpy as np
import pytest

from humpy.correlation import (
    PolychoricCorrelation,
    PolyserialCorrelation,
    adhoc_polyserial,
    f_hat,
    npn_pearson,
    spearman,
)


# ---------------------------------------------------------------------------
# Fixtures / shared data helpers
# ---------------------------------------------------------------------------


def _make_latent_pair(rho: float = 0.6, n: int = 500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return two correlated binary arrays drawn from a latent Gaussian with correlation *rho*."""
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n)
    x = (z[:, 0] > 0).astype(int)
    y = (z[:, 1] > 0).astype(int)
    return x, y


def _make_mixed_pair(rho: float = 0.6, n: int = 500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (continuous, ordinal) pair with latent correlation *rho*."""
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n)
    x_cont = z[:, 0]
    y_ord = (z[:, 1] > -0.5).astype(int) + (z[:, 1] > 0.5).astype(int)
    return x_cont, y_ord


def _make_ordinal_pair(n: int, rho: float, n_cats: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Draw two balanced ordinal variables from a latent bivariate Gaussian."""
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n)
    cuts = np.linspace(0, 100, n_cats + 1)[1:-1]
    x = np.searchsorted(np.percentile(z[:, 0], cuts), z[:, 0]).astype(float)
    y = np.searchsorted(np.percentile(z[:, 1], cuts), z[:, 1]).astype(float)
    return x, y


# ---------------------------------------------------------------------------
# Legacy function tests — kept exactly as before
# ---------------------------------------------------------------------------


class TestSpearman:
    """Tests for Spearman's rho correlation."""

    def test_perfect_positive_correlation(self) -> None:
        """Test perfect positive correlation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        assert spearman(x, y) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self) -> None:
        """Test perfect negative correlation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])
        assert spearman(x, y) == pytest.approx(-1.0)

    def test_no_correlation(self) -> None:
        """Test no correlation (approximately)."""
        np.random.seed(42)
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        # Should be close to 0 with random data
        assert abs(spearman(x, y)) < 0.1

    def test_monotonic_relationship(self) -> None:
        """Test correlation for monotonic non-linear relationship."""
        x = np.array([1, 2, 3, 4, 5])
        y = x**2  # Monotonic but non-linear
        assert spearman(x, y) == pytest.approx(1.0)


class TestFHat:
    """Tests for nonparanormal transformation."""

    def test_output_shape(self) -> None:
        """Test that output has same shape as input."""
        x = np.random.randn(100)
        result = f_hat(x)
        assert result.shape == x.shape

    def test_unit_variance(self) -> None:
        """Test that output has approximately unit variance."""
        x = np.random.randn(1000)
        result = f_hat(x)
        assert np.std(result, ddof=1) == pytest.approx(1.0, rel=0.1)

    def test_preserves_order(self) -> None:
        """Test that transformation preserves order."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = f_hat(x)
        # Differences should all be positive
        assert np.all(np.diff(result) > 0)


class TestNpnPearson:
    """Tests for nonparanormal Pearson correlation."""

    def test_continuous_discrete_correlation(self) -> None:
        """Test correlation between continuous and discrete."""
        np.random.seed(42)
        # Create correlated data
        n = 200
        cont = np.random.randn(n)
        # Discrete is related to continuous
        disc = (cont > 0).astype(int) + (cont > 1).astype(int)

        corr = npn_pearson(cont, disc)
        # Should be positive since disc increases with cont
        assert corr > 0.3

    def test_returns_scalar(self) -> None:
        """Test that result is a scalar."""
        x = np.random.randn(50)
        y = np.random.randint(0, 3, 50)
        result = npn_pearson(x, y)
        assert isinstance(result, float)


class TestAdhocPolyserial:
    """Tests for adhoc polyserial correlation."""

    def test_both_continuous(self) -> None:
        """Test with two continuous variables."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = x + np.random.randn(100) * 0.5

        corr = adhoc_polyserial(x, y, n_levels_threshold=20)
        assert 0.5 < corr < 1.0

    def test_both_discrete(self) -> None:
        """Test with two discrete variables."""
        np.random.seed(42)
        x = np.random.randint(0, 5, 100)
        y = x + np.random.randint(-1, 2, 100)
        y = np.clip(y, 0, 5)

        corr = adhoc_polyserial(x, y, n_levels_threshold=20)
        assert corr > 0  # Should be positive

    def test_mixed_continuous_discrete(self) -> None:
        """Test with one continuous and one discrete variable."""
        np.random.seed(42)
        n = 200
        cont = np.random.randn(n)
        # Create discrete variable related to continuous
        disc = (cont > -1).astype(int) + (cont > 0).astype(int) + (cont > 1).astype(int)

        corr = adhoc_polyserial(cont, disc, n_levels_threshold=20)
        assert corr > 0.3

    def test_correlation_bounds(self) -> None:
        """Test that correlation is within [-1, 1]."""
        np.random.seed(42)
        for _ in range(10):
            x = np.random.randn(50)
            y = np.random.randint(0, 3, 50)
            corr = adhoc_polyserial(x, y)
            assert -1 <= corr <= 1

    def test_max_cor_clipping(self) -> None:
        """Test that max_cor parameter clips extreme correlations."""
        np.random.seed(42)
        # Create highly correlated data
        x = np.arange(50).astype(float)
        y = np.arange(50) // 10  # Discrete version, perfectly correlated

        corr = adhoc_polyserial(x, y, max_cor=0.99, n_levels_threshold=20)
        assert abs(corr) <= 0.99


# ---------------------------------------------------------------------------
# CorrelationMeasure ABC — base-class contracts
# ---------------------------------------------------------------------------


class TestCorrelationMeasureBase:
    """Tests for invariants that every CorrelationMeasure subclass must satisfy."""

    @pytest.mark.parametrize("cls", [PolychoricCorrelation, PolyserialCorrelation])
    def test_invalid_max_cor_above_one_raises(self, cls) -> None:
        with pytest.raises(ValueError, match="max_cor"):
            cls(max_cor=1.5)

    @pytest.mark.parametrize("cls", [PolychoricCorrelation, PolyserialCorrelation])
    def test_invalid_max_cor_zero_raises(self, cls) -> None:
        with pytest.raises(ValueError, match="max_cor"):
            cls(max_cor=0.0)

    @pytest.mark.parametrize("cls", [PolychoricCorrelation, PolyserialCorrelation])
    def test_correlation_before_fit_raises(self, cls) -> None:
        """Accessing .correlation before .fit should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="fit"):
            _ = cls().correlation

    def test_polychoric_fit_returns_self(self) -> None:
        """fit() must return the same instance to enable method chaining."""
        x, y = _make_latent_pair()
        instance = PolychoricCorrelation()
        assert instance.fit(x, y) is instance

    def test_polyserial_fit_returns_self(self) -> None:
        cont, ord_var = _make_mixed_pair()
        instance = PolyserialCorrelation()
        assert instance.fit(cont, ord_var) is instance

    def test_polychoric_method_chaining(self) -> None:
        x, y = _make_latent_pair()
        rho = PolychoricCorrelation().fit(x, y).correlation
        assert isinstance(rho, float)

    def test_polyserial_method_chaining(self) -> None:
        cont, ord_var = _make_mixed_pair()
        rho = PolyserialCorrelation().fit(cont, ord_var).correlation
        assert isinstance(rho, float)


# ---------------------------------------------------------------------------
# Shared input-validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Input-sanitation checks exercised through the public class interface."""

    def _polychoric_pair(self) -> tuple[np.ndarray, np.ndarray]:
        return _make_latent_pair()

    def _polyserial_pair(self) -> tuple[np.ndarray, np.ndarray]:
        return _make_mixed_pair()

    @pytest.mark.parametrize(
        "bad_x, bad_y",
        [
            (np.array([]), np.array([])),
            (np.array([1.0]), np.array([1.0])),  # below min_samples=5
        ],
    )
    def test_polychoric_empty_or_too_short_raises(self, bad_x, bad_y) -> None:
        with pytest.raises(ValueError):
            PolychoricCorrelation().fit(bad_x, bad_y)

    def test_polychoric_length_mismatch_raises(self) -> None:
        x = np.array([0, 1, 0, 1, 0])
        y = np.array([0, 1, 0, 1])
        with pytest.raises(ValueError, match="same length"):
            PolychoricCorrelation().fit(x, y)

    def test_polychoric_constant_array_raises(self) -> None:
        x = np.zeros(50)
        y = np.ones(50)
        with pytest.raises(ValueError, match="constant"):
            PolychoricCorrelation().fit(x, y)

    def test_polychoric_all_nan_raises(self) -> None:
        x = np.full(20, np.nan)
        y = np.random.randint(0, 2, 20).astype(float)
        with pytest.raises(ValueError):
            PolychoricCorrelation().fit(x, y)

    def test_polychoric_infinite_ordinal_raises(self) -> None:
        x = np.array([0.0, 1.0, np.inf, 0.0, 1.0] * 10)
        y = np.array([1.0, 0.0, 1.0, 0.0, 1.0] * 10)
        with pytest.raises(ValueError, match="infinite"):
            PolychoricCorrelation().fit(x, y)

    def test_polychoric_single_category_raises(self) -> None:
        """Ordinal variable with only one unique value must be rejected."""
        x = np.zeros(30)  # all same — but also constant, hits that check first
        y = np.array([0] * 15 + [1] * 15, dtype=float)
        with pytest.raises(ValueError):
            PolychoricCorrelation().fit(x, y)

    def test_polyserial_2d_input_raises(self) -> None:
        x = np.ones((10, 2))
        y = np.ones(10)
        with pytest.raises(ValueError, match="1-D"):
            PolyserialCorrelation().fit(x, y)

    def test_polyserial_infinite_continuous_raises(self) -> None:
        cont = np.random.randn(50)
        cont[5] = np.inf
        ord_var = np.random.randint(0, 3, 50).astype(float)
        with pytest.raises(ValueError, match="infinite"):
            PolyserialCorrelation().fit(cont, ord_var)

    def test_polyserial_both_continuous_raises(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.standard_normal(100)
        y = rng.standard_normal(100)
        with pytest.raises(ValueError, match="continuous"):
            PolyserialCorrelation(n_levels_threshold=5).fit(x, y)

    def test_polyserial_both_ordinal_raises(self) -> None:
        x = np.random.randint(0, 4, 100).astype(float)
        y = np.random.randint(0, 4, 100).astype(float)
        with pytest.raises(ValueError, match="ordinal|PolychoricCorrelation"):
            PolyserialCorrelation().fit(x, y)

    def test_polyserial_invalid_n_levels_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="n_levels_threshold"):
            PolyserialCorrelation(n_levels_threshold=1)


# ---------------------------------------------------------------------------
# PolychoricCorrelation — behaviour tests
# ---------------------------------------------------------------------------


class TestPolychoricCorrelation:
    """Behavioural tests for PolychoricCorrelation."""

    def test_positive_latent_correlation_recovered(self) -> None:
        """Estimate should be positive when latent correlation is positive."""
        x, y = _make_latent_pair(rho=0.6, n=500)
        rho = PolychoricCorrelation().fit(x, y).correlation
        assert rho > 0.3

    def test_negative_latent_correlation_recovered(self) -> None:
        """Estimate should be negative when latent correlation is negative."""
        x, y = _make_latent_pair(rho=-0.6, n=500)
        rho = PolychoricCorrelation().fit(x, y).correlation
        assert rho < -0.3

    def test_result_within_bounds(self) -> None:
        x, y = _make_latent_pair(rho=0.4)
        rho = PolychoricCorrelation().fit(x, y).correlation
        assert -1.0 <= rho <= 1.0

    def test_custom_max_cor_clips(self) -> None:
        """A tight max_cor should clip the estimate."""
        x, y = _make_latent_pair(rho=0.9, n=2000)
        rho = PolychoricCorrelation(max_cor=0.5).fit(x, y).correlation
        assert abs(rho) <= 0.5

    def test_result_is_float(self) -> None:
        x, y = _make_latent_pair()
        rho = PolychoricCorrelation().fit(x, y).correlation
        assert isinstance(rho, float)

    def test_multi_category_ordinal(self) -> None:
        """Should work with more than 2 ordinal levels."""
        rng = np.random.default_rng(7)
        z = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=300)
        x = np.digitize(z[:, 0], bins=[-1.0, 0.0, 1.0])
        y = np.digitize(z[:, 1], bins=[-1.0, 0.0, 1.0])
        rho = PolychoricCorrelation().fit(x, y).correlation
        assert rho > 0.2

    def test_symmetry(self) -> None:
        """Swapping x and y must yield the same correlation."""
        x, y = _make_latent_pair(rho=0.5)
        rho_xy = PolychoricCorrelation().fit(x, y).correlation
        rho_yx = PolychoricCorrelation().fit(y, x).correlation
        assert rho_xy == pytest.approx(rho_yx, abs=1e-10)

    @pytest.mark.parametrize(
        "rho_true, n_cats",
        [
            (0.0, 2),
            (0.5, 2),
            (-0.5, 2),
            (0.5, 4),
            (0.7, 7),
        ],
    )
    def test_newton_and_brent_agree(self, rho_true: float, n_cats: int) -> None:
        """Both solvers must converge to the same MLE (within numerical tolerance)."""
        x, y = _make_ordinal_pair(n=500, rho=rho_true, n_cats=n_cats, seed=99)
        rho_newton = PolychoricCorrelation(solver="newton").fit(x, y).correlation
        rho_brent = PolychoricCorrelation(solver="brent").fit(x, y).correlation
        assert rho_newton == pytest.approx(rho_brent, abs=1e-4)


# ---------------------------------------------------------------------------
# PolyserialCorrelation — behaviour tests
# ---------------------------------------------------------------------------


class TestPolyserialCorrelation:
    """Behavioural tests for PolyserialCorrelation."""

    def test_positive_latent_correlation_recovered(self) -> None:
        cont, ord_var = _make_mixed_pair(rho=0.6, n=500)
        rho = PolyserialCorrelation().fit(cont, ord_var).correlation
        assert rho > 0.3

    def test_negative_latent_correlation_recovered(self) -> None:
        cont, ord_var = _make_mixed_pair(rho=-0.6, n=500)
        rho = PolyserialCorrelation().fit(cont, ord_var).correlation
        assert rho < -0.3

    def test_result_within_bounds(self) -> None:
        cont, ord_var = _make_mixed_pair(rho=0.4)
        rho = PolyserialCorrelation().fit(cont, ord_var).correlation
        assert -1.0 <= rho <= 1.0

    def test_custom_max_cor_clips(self) -> None:
        cont, ord_var = _make_mixed_pair(rho=0.9, n=2000)
        rho = PolyserialCorrelation(max_cor=0.5).fit(cont, ord_var).correlation
        assert abs(rho) <= 0.5

    def test_result_is_float(self) -> None:
        cont, ord_var = _make_mixed_pair()
        rho = PolyserialCorrelation().fit(cont, ord_var).correlation
        assert isinstance(rho, float)

    def test_auto_detects_which_variable_is_ordinal(self) -> None:
        """fit(cont, ord) and fit(ord, cont) should give the same result."""
        cont, ord_var = _make_mixed_pair(rho=0.5, n=400)
        rho_fwd = PolyserialCorrelation().fit(cont, ord_var).correlation
        rho_rev = PolyserialCorrelation().fit(ord_var, cont).correlation
        assert rho_fwd == pytest.approx(rho_rev, abs=1e-10)

    def test_second_fit_overwrites_first(self) -> None:
        """Calling fit() twice should update .correlation, not accumulate."""
        cont_a, ord_a = _make_mixed_pair(rho=0.7, n=400, seed=1)
        cont_b, ord_b = _make_mixed_pair(rho=-0.7, n=400, seed=2)
        est = PolyserialCorrelation()
        est.fit(cont_a, ord_a)
        rho_first = est.correlation
        est.fit(cont_b, ord_b)
        rho_second = est.correlation
        # The two estimates should differ in sign
        assert rho_first > 0
        assert rho_second < 0


# ---------------------------------------------------------------------------
# Consistency: classes produce the same results as the legacy dispatcher
# ---------------------------------------------------------------------------


class TestConsistencyWithLegacy:
    """Verify that the class API and adhoc_polyserial are numerically identical."""

    def test_polychoric_class_matches_dispatcher_ordinal_pair(self) -> None:
        x, y = _make_latent_pair(rho=0.5, n=300, seed=10)
        via_class = PolychoricCorrelation().fit(x, y).correlation
        via_func = adhoc_polyserial(x, y, n_levels_threshold=20)
        assert via_class == pytest.approx(via_func, abs=1e-12)

    def test_polyserial_class_matches_dispatcher_mixed_pair(self) -> None:
        cont, ord_var = _make_mixed_pair(rho=0.5, n=300, seed=11)
        via_class = PolyserialCorrelation().fit(cont, ord_var).correlation
        via_func = adhoc_polyserial(cont, ord_var, n_levels_threshold=20)
        assert via_class == pytest.approx(via_func, abs=1e-12)

    def test_polychoric_max_cor_respected_identically(self) -> None:
        x, y = _make_latent_pair(rho=0.9, n=1000, seed=20)
        max_cor = 0.7
        via_class = PolychoricCorrelation(max_cor=max_cor).fit(x, y).correlation
        via_func = adhoc_polyserial(x, y, max_cor=max_cor, n_levels_threshold=20)
        assert via_class == pytest.approx(via_func, abs=1e-12)

    def test_polyserial_max_cor_respected_identically(self) -> None:
        cont, ord_var = _make_mixed_pair(rho=0.9, n=1000, seed=21)
        max_cor = 0.7
        via_class = PolyserialCorrelation(max_cor=max_cor).fit(cont, ord_var).correlation
        via_func = adhoc_polyserial(cont, ord_var, max_cor=max_cor, n_levels_threshold=20)
        assert via_class == pytest.approx(via_func, abs=1e-12)
