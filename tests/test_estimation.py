"""Tests for hume graph estimation functions."""

import numpy as np
import pandas as pd
import pytest

from hume.estimation import (
    MixedGraphResult,
    edgenumber,
    mixed_graph_gauss,
    mixed_graph_nonpara,
)


class TestEdgenumber:
    """Tests for edgenumber function."""

    def test_empty_graph(self) -> None:
        """Test edge count for identity matrix (no edges)."""
        precision = np.eye(5)
        assert edgenumber(precision) == 0

    def test_full_graph(self) -> None:
        """Test edge count for fully connected graph."""
        precision = np.ones((5, 5))
        # Lower triangular count: 5*4/2 = 10 edges
        assert edgenumber(precision) == 10

    def test_sparse_graph(self) -> None:
        """Test edge count for sparse graph."""
        precision = np.eye(4)
        precision[0, 1] = precision[1, 0] = 0.5
        precision[2, 3] = precision[3, 2] = 0.3
        assert edgenumber(precision) == 2

    def test_threshold(self) -> None:
        """Test thresholding with cut parameter."""
        precision = np.eye(3)
        precision[0, 1] = precision[1, 0] = 0.3
        precision[0, 2] = precision[2, 0] = 0.5

        # Without threshold
        assert edgenumber(precision) == 2

        # With threshold
        assert edgenumber(precision, cut=0.4) == 1


class TestMixedGraphResult:
    """Tests for MixedGraphResult dataclass."""

    def test_repr(self) -> None:
        """Test string representation."""
        result = MixedGraphResult(
            precision_matrix=np.eye(5),
            adjacency_matrix=np.eye(5, dtype=bool),
            correlation_matrix=np.eye(5),
            n_edges=3,
            max_degree=2,
            initial_mat_singular=False,
        )
        assert "n_nodes=5" in repr(result)
        assert "n_edges=3" in repr(result)


class TestMixedGraphGauss:
    """Tests for mixed_graph_gauss function."""

    @pytest.fixture
    def continuous_data(self) -> pd.DataFrame:
        """Create continuous test data."""
        np.random.seed(42)
        n = 100
        d = 5
        return pd.DataFrame(np.random.randn(n, d))

    @pytest.fixture
    def mixed_data(self) -> pd.DataFrame:
        """Create mixed continuous/discrete test data."""
        np.random.seed(42)
        n = 100
        continuous = np.random.randn(n, 3)
        discrete = np.random.randint(0, 3, (n, 2))
        data = np.hstack([continuous, discrete])
        return pd.DataFrame(data, columns=[f"var_{i}" for i in range(5)])

    def test_returns_result_object(self, mixed_data: pd.DataFrame) -> None:
        """Test that function returns MixedGraphResult."""
        result = mixed_graph_gauss(mixed_data, verbose=False)
        assert isinstance(result, MixedGraphResult)

    def test_precision_matrix_shape(self, mixed_data: pd.DataFrame) -> None:
        """Test precision matrix has correct shape."""
        result = mixed_graph_gauss(mixed_data, verbose=False)
        d = mixed_data.shape[1]
        assert result.precision_matrix.shape == (d, d)

    def test_adjacency_matrix_symmetric(self, mixed_data: pd.DataFrame) -> None:
        """Test adjacency matrix is symmetric."""
        result = mixed_graph_gauss(mixed_data, verbose=False)
        assert np.allclose(result.adjacency_matrix, result.adjacency_matrix.T)

    def test_feature_names_from_dataframe(self, mixed_data: pd.DataFrame) -> None:
        """Test feature names are extracted from DataFrame."""
        result = mixed_graph_gauss(mixed_data, verbose=False)
        assert result.feature_names == [f"var_{i}" for i in range(5)]

    def test_custom_feature_names(self, mixed_data: pd.DataFrame) -> None:
        """Test custom feature names."""
        names = ["a", "b", "c", "d", "e"]
        result = mixed_graph_gauss(mixed_data, verbose=False, feature_names=names)
        assert result.feature_names == names

    def test_accepts_numpy_array(self) -> None:
        """Test function accepts numpy array input."""
        np.random.seed(42)
        data = np.random.randn(50, 4)
        result = mixed_graph_gauss(data, verbose=False)
        assert result.precision_matrix.shape == (4, 4)


class TestMixedGraphNonpara:
    """Tests for mixed_graph_nonpara function."""

    @pytest.fixture
    def mixed_data(self) -> pd.DataFrame:
        """Create mixed continuous/discrete test data."""
        np.random.seed(42)
        n = 100
        continuous = np.random.randn(n, 3)
        discrete = np.random.randint(0, 3, (n, 2))
        data = np.hstack([continuous, discrete])
        return pd.DataFrame(data, columns=[f"var_{i}" for i in range(5)])

    def test_returns_result_object(self, mixed_data: pd.DataFrame) -> None:
        """Test that function returns MixedGraphResult."""
        result = mixed_graph_nonpara(mixed_data, verbose=False)
        assert isinstance(result, MixedGraphResult)

    def test_precision_matrix_shape(self, mixed_data: pd.DataFrame) -> None:
        """Test precision matrix has correct shape."""
        result = mixed_graph_nonpara(mixed_data, verbose=False)
        d = mixed_data.shape[1]
        assert result.precision_matrix.shape == (d, d)

    def test_adjacency_matrix_symmetric(self, mixed_data: pd.DataFrame) -> None:
        """Test adjacency matrix is symmetric."""
        result = mixed_graph_nonpara(mixed_data, verbose=False)
        assert np.allclose(result.adjacency_matrix, result.adjacency_matrix.T)

    def test_n_edges_non_negative(self, mixed_data: pd.DataFrame) -> None:
        """Test number of edges is non-negative."""
        result = mixed_graph_nonpara(mixed_data, verbose=False)
        assert result.n_edges >= 0

    def test_max_degree_valid(self, mixed_data: pd.DataFrame) -> None:
        """Test max degree is valid."""
        result = mixed_graph_nonpara(mixed_data, verbose=False)
        d = mixed_data.shape[1]
        assert 0 <= result.max_degree < d

    def test_param_effect(self, mixed_data: pd.DataFrame) -> None:
        """Test that param affects sparsity."""
        result_sparse = mixed_graph_nonpara(mixed_data, verbose=False, param=0.5)
        result_dense = mixed_graph_nonpara(mixed_data, verbose=False, param=0.0)
        # Higher param should lead to sparser graph
        assert result_sparse.n_edges <= result_dense.n_edges


class TestIntegration:
    """Integration tests for the full estimation pipeline."""

    def test_known_structure(self) -> None:
        """Test recovery of known sparse structure."""
        np.random.seed(42)
        n = 500

        # Create data with known correlation structure
        # Variables 0 and 1 are correlated, 2 and 3 are independent
        x0 = np.random.randn(n)
        x1 = 0.8 * x0 + 0.6 * np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)

        data = pd.DataFrame({
            "x0": x0,
            "x1": x1,
            "x2": x2,
            "x3": x3,
        })

        result = mixed_graph_nonpara(data, verbose=False, param=0.1)

        # Check that x0-x1 edge exists
        assert result.adjacency_matrix[0, 1] or result.adjacency_matrix[1, 0]

    def test_binary_variables(self) -> None:
        """Test with binary discrete variables."""
        np.random.seed(42)
        n = 200

        continuous = np.random.randn(n, 2)
        binary = np.random.binomial(1, 0.5, (n, 2))

        data = pd.DataFrame(np.hstack([continuous, binary]), columns=["cont1", "cont2", "bin1", "bin2"])

        result = mixed_graph_nonpara(data, verbose=False)
        assert result.precision_matrix.shape == (4, 4)

    def test_ordinal_variables(self) -> None:
        """Test with ordinal discrete variables."""
        np.random.seed(42)
        n = 200

        continuous = np.random.randn(n, 2)
        ordinal = np.random.randint(0, 5, (n, 2))  # 5 levels

        data = pd.DataFrame(np.hstack([continuous, ordinal]), columns=["cont1", "cont2", "ord1", "ord2"])

        result = mixed_graph_nonpara(data, verbose=False, n_levels_threshold=20)
        assert result.precision_matrix.shape == (4, 4)
