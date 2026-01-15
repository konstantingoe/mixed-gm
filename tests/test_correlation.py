"""Tests for hume correlation functions."""

import numpy as np
import pytest

from hume.correlation import adhoc_polyserial, f_hat, npn_pearson, spearman


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
