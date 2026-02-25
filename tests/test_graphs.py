"""Tests for humpy/graphs.py."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from humpy.graphs import GRAPH, UGRAPH


class TestUGRAPH:
    """Test class for UGRAPH."""

    @pytest.fixture(scope="class")
    def example_ugraph(self) -> UGRAPH:
        """UGRAPH with three nodes and two edges: A-B, A-C."""
        return UGRAPH(nodes=["A", "B", "C"], edges=[("A", "B"), ("A", "C")])

    # ------ construction ---------------------------------------------------

    def test_instance_is_created(self) -> None:
        """UGRAPH is a GRAPH subclass."""
        ug = UGRAPH(nodes=["A", "B", "C"])
        assert isinstance(ug, UGRAPH)
        assert isinstance(ug, GRAPH)

    def test_empty_graph(self) -> None:
        """Default constructor yields empty graph."""
        ug = UGRAPH()
        assert ug.num_nodes == 0
        assert ug.num_edges == 0

    def test_nodes_and_edges_populated(self, example_ugraph: UGRAPH) -> None:
        """Nodes and edges are set correctly on construction."""
        assert set(example_ugraph.nodes) == {"A", "B", "C"}
        assert example_ugraph.num_nodes == 3
        assert example_ugraph.num_edges == 2

    def test_edges_are_stored(self, example_ugraph: UGRAPH) -> None:
        """Both edges are present."""
        edge_set = {frozenset(e) for e in example_ugraph.edges}
        assert frozenset(("A", "B")) in edge_set
        assert frozenset(("A", "C")) in edge_set

    def test_adding_edge_creates_nodes(self) -> None:
        """Nodes are created implicitly when an edge is added."""
        ug = UGRAPH(edges=[("X", "Y")])
        assert "X" in ug.nodes
        assert "Y" in ug.nodes

    # ------ neighbors / adjacency -----------------------------------------

    def test_neighbors(self, example_ugraph: UGRAPH) -> None:
        """neighbors() returns correct neighbour sets."""
        assert example_ugraph.neighbors("A") == {"B", "C"}
        assert example_ugraph.neighbors("B") == {"A"}
        assert example_ugraph.neighbors("C") == {"A"}

    def test_neighbors_isolated_node(self) -> None:
        """Isolated node has empty neighbour set."""
        ug = UGRAPH(nodes=["A", "B"])
        assert ug.neighbors("A") == set()

    def test_is_adjacent_true(self, example_ugraph: UGRAPH) -> None:
        """is_adjacent returns True for connected nodes in both orderings."""
        assert example_ugraph.is_adjacent("A", "B")
        assert example_ugraph.is_adjacent("B", "A")

    def test_is_adjacent_false(self, example_ugraph: UGRAPH) -> None:
        """is_adjacent returns False for non-connected nodes."""
        assert not example_ugraph.is_adjacent("B", "C")

    # ------ is_clique ------------------------------------------------------

    def test_is_clique_true(self) -> None:
        """Triangle graph: all three nodes form a clique."""
        ug = UGRAPH(edges=[("A", "B"), ("B", "C"), ("A", "C")])
        assert ug.is_clique({"A", "B", "C"})

    def test_is_clique_false(self, example_ugraph: UGRAPH) -> None:
        """Path graph A-B-C: {A,B,C} is not a clique (B-C missing)."""
        ug = UGRAPH(edges=[("A", "B"), ("B", "C")])
        assert not ug.is_clique({"A", "B", "C"})

    def test_is_clique_single_node(self, example_ugraph: UGRAPH) -> None:
        """A single-node set is trivially a clique."""
        assert example_ugraph.is_clique({"A"})

    # ------ from_pandas_adjacency -----------------------------------------

    def test_from_pandas_adjacency(self, example_ugraph: UGRAPH) -> None:
        """Round-trip through adjacency matrix preserves structure."""
        amat = pd.DataFrame(
            [[0, 1, 1], [1, 0, 0], [1, 0, 0]],
            columns=["A", "B", "C"],
            index=["A", "B", "C"],
        )
        ug = UGRAPH.from_pandas_adjacency(pd_amat=amat)
        assert set(ug.nodes) == {"A", "B", "C"}
        assert ug.num_edges == 2
        assert ug.is_adjacent("A", "B")
        assert ug.is_adjacent("A", "C")
        assert not ug.is_adjacent("B", "C")

    def test_from_pandas_adjacency_deduplicates(self) -> None:
        """from_pandas_adjacency does not double-count symmetric entries."""
        amat = pd.DataFrame(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            columns=["A", "B", "C"],
            index=["A", "B", "C"],
        )
        ug = UGRAPH.from_pandas_adjacency(pd_amat=amat)
        assert ug.num_edges == 2

    # ------ remove_edge ---------------------------------------------------

    def test_remove_edge(self) -> None:
        """remove_edge deletes the edge and updates neighbours."""
        ug = UGRAPH(edges=[("A", "B"), ("A", "C")])
        ug.remove_edge("A", "B")
        assert not ug.is_adjacent("A", "B")
        assert "B" not in ug.neighbors("A")
        assert "A" not in ug.neighbors("B")

    def test_remove_edge_both_orderings(self) -> None:
        """remove_edge works regardless of argument order."""
        ug = UGRAPH(edges=[("A", "B")])
        ug.remove_edge("B", "A")
        assert not ug.is_adjacent("A", "B")

    def test_remove_nonexistent_edge_raises(self) -> None:
        """Removing a missing edge raises AssertionError."""
        ug = UGRAPH(edges=[("A", "B")])
        with pytest.raises(AssertionError, match="Edge does not exist in current UGRAPH"):
            ug.remove_edge("A", "C")

    # ------ remove_node ---------------------------------------------------

    def test_remove_node(self) -> None:
        """remove_node deletes the node and all incident edges."""
        ug = UGRAPH(edges=[("A", "B"), ("A", "C")])
        ug.remove_node("A")
        assert "A" not in ug.nodes
        assert not ug.is_adjacent("A", "B")
        assert not ug.is_adjacent("A", "C")
        assert "A" not in ug.neighbors("B")
        assert "A" not in ug.neighbors("C")

    # ------ adjacency_matrix ----------------------------------------------

    def test_adjacency_matrix_shape(self, example_ugraph: UGRAPH) -> None:
        """Adjacency matrix has the right shape."""
        amat = example_ugraph.adjacency_matrix
        d = example_ugraph.num_nodes
        assert amat.shape == (d, d)

    def test_adjacency_matrix_is_symmetric(self, example_ugraph: UGRAPH) -> None:
        """Undirected graph has a symmetric adjacency matrix."""
        amat = example_ugraph.adjacency_matrix.to_numpy()
        assert np.allclose(amat, amat.T)

    def test_adjacency_matrix_entry_sum(self, example_ugraph: UGRAPH) -> None:
        """Sum of entries equals 2 * num_edges (symmetric matrix)."""
        amat = example_ugraph.adjacency_matrix
        assert int(amat.to_numpy().sum()) == 2 * example_ugraph.num_edges

    def test_adjacency_matrix_zero_diagonal(self, example_ugraph: UGRAPH) -> None:
        """No self-loops: diagonal is all zeros."""
        amat = example_ugraph.adjacency_matrix.to_numpy()
        assert np.all(np.diag(amat) == 0)

    # ------ causal_order --------------------------------------------------

    def test_causal_order_is_none(self, example_ugraph: UGRAPH) -> None:
        """Undirected graphs have no causal order."""
        assert example_ugraph.causal_order is None

    # ------ copy ----------------------------------------------------------

    def test_copy_is_independent(self, example_ugraph: UGRAPH) -> None:
        """copy() returns a new independent UGRAPH."""
        ug_copy = example_ugraph.copy()
        assert isinstance(ug_copy, UGRAPH)
        assert set(ug_copy.nodes) == set(example_ugraph.nodes)
        assert ug_copy.num_edges == example_ugraph.num_edges

        # Mutating the copy does not affect the original
        ug_copy.remove_edge("A", "B")
        assert example_ugraph.is_adjacent("A", "B")

    # ------ to_networkx ---------------------------------------------------

    def test_to_networkx(self, example_ugraph: UGRAPH) -> None:
        """to_networkx produces an equivalent nx.Graph."""
        nxg = example_ugraph.to_networkx()
        assert isinstance(nxg, nx.Graph)
        assert set(nxg.nodes) == set(example_ugraph.nodes)
        edge_set = {frozenset(e) for e in nxg.edges}
        for e in example_ugraph.edges:
            assert frozenset(e) in edge_set
