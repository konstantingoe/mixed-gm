"""Test for hume/graphs.py."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from hume.graphs import DAG, GRAPH, PDAG, UGRAPH, LayeredDAG, dag2cpdag


class TestPDAG:
    """Test PDAG class."""

    @pytest.fixture(scope="class")
    def mixed_pdag(self) -> PDAG:
        """Initite PDAG with directed and undirected edges.

        Returns:
            PDAG: Resulting PDAG.
        """
        pdag = PDAG(
            nodes=["A", "B", "C"],
            dir_edges=[("A", "B"), ("A", "C")],
            undir_edges=[("B", "C")],
        )
        return pdag

    def test_instance_is_created(self):
        """Test whether instance is created."""
        pdag = PDAG(nodes=["A", "B", "C"])
        assert isinstance(pdag, PDAG)
        assert isinstance(pdag, GRAPH)

    def test_dir_edges(self):
        """Test directed edges."""
        pdag = PDAG(nodes=["A", "B", "C"], dir_edges=[("A", "B"), ("A", "C")])
        num_edges = len([("A", "B"), ("A", "C")])
        assert pdag.num_dir_edges == num_edges
        assert pdag.num_undir_edges == 0
        assert set(pdag.dir_edges) == {("A", "B"), ("A", "C")}

    def test_undir_edges(self):
        """Test undirected edges."""
        pdag = PDAG(nodes=["A", "B", "C"], undir_edges=[("A", "B"), ("A", "C")])
        num_edges = len([("A", "B"), ("A", "C")])
        assert pdag.num_dir_edges == 0
        assert pdag.num_undir_edges == num_edges
        assert set(pdag.undir_edges) == {("A", "B"), ("A", "C")}

    def test_mixed_edges(self, mixed_pdag: PDAG):
        """Test both edges being present.

        Args:
            mixed_pdag (PDAG): _description_
        """
        num_edges = 2
        assert mixed_pdag.num_dir_edges == num_edges
        assert mixed_pdag.num_undir_edges == 1
        assert set(mixed_pdag.dir_edges) == {("A", "B"), ("A", "C")}
        assert set(mixed_pdag.undir_edges) == {("B", "C")}

    def test_children(self, mixed_pdag: PDAG):
        """Test children correctly assigned.

        Args:
            mixed_pdag (PDAG): _description_
        """
        assert mixed_pdag.children(node="A") == {"B", "C"}
        assert mixed_pdag.children(node="B") == set()
        assert mixed_pdag.children(node="C") == set()

    def test_parents(self, mixed_pdag: PDAG):
        """Test parents correctly assigned.

        Args:
            mixed_pdag (PDAG): _description_
        """
        assert mixed_pdag.parents(node="A") == set()
        assert mixed_pdag.parents(node="B") == {"A"}
        assert mixed_pdag.parents(node="C") == {"A"}

    def test_neighbors(self, mixed_pdag: PDAG):
        """Test neighbors correctly assigned.

        Args:
            mixed_pdag (PDAG): _description_
        """
        assert mixed_pdag.neighbors(node="C") == {"B", "A"}
        assert mixed_pdag.undir_neighbors(node="C") == {"B"}
        assert mixed_pdag.is_adjacent(i="B", j="C")

    def test_from_pandas_adjacency(self, mixed_pdag: PDAG):
        """Test from_pandas_adjacency works.

        Args:
            mixed_pdag (PDAG): _description_
        """
        amat = pd.DataFrame(
            [[0, 1, 1], [0, 0, 1], [0, 1, 0]],
            columns=["A", "B", "C"],
            index=["A", "B", "C"],
        )
        from_pandas_pdag = PDAG.from_pandas_adjacency(pd_amat=amat)
        assert np.allclose(from_pandas_pdag.adjacency_matrix, amat)
        assert set(from_pandas_pdag.dir_edges) == set(mixed_pdag.dir_edges)
        assert from_pandas_pdag.num_undir_edges == mixed_pdag.num_undir_edges

    def test_remove_edge(self, mixed_pdag: PDAG):
        """Test removing edges works.

        Args:
            mixed_pdag (PDAG): _description_
        """
        assert ("A", "C") in mixed_pdag.dir_edges
        mixed_pdag.remove_edge("A", "C")
        assert ("A", "C") not in mixed_pdag.dir_edges
        with pytest.raises(AssertionError, match=r"Edge does not exist in current PDAG"):
            mixed_pdag.remove_edge("B", "A")

    def test_change_undir_edge_to_dir_edge(self, mixed_pdag: PDAG):
        """Test orient undirected edge.

        Args:
            mixed_pdag (PDAG): _description_
        """
        assert ("B", "C") in mixed_pdag.undir_edges or (
            "C",
            "B",
        ) in mixed_pdag.undir_edges
        mixed_pdag.undir_to_dir_edge(tail="C", head="B")
        assert ("C", "B") in mixed_pdag.dir_edges
        assert ("B", "C") not in mixed_pdag.dir_edges
        assert ("B", "C") not in mixed_pdag.undir_edges
        assert ("C", "B") not in mixed_pdag.undir_edges

    def test_remove_node(self, mixed_pdag: PDAG):
        """Test remove node from graph.

        Args:
            mixed_pdag (PDAG): _description_
        """
        assert "C" in mixed_pdag.nodes
        mixed_pdag.remove_node("C")
        assert "C" not in mixed_pdag.nodes

    def test_to_dag(self, mixed_pdag: PDAG):
        """Test transform PDAG to DAG.

        Args:
            mixed_pdag (PDAG): _description_
        """
        dag = mixed_pdag.to_dag()
        assert nx.is_directed_acyclic_graph(dag)
        assert set(mixed_pdag.dir_edges).issubset(set(dag.edges))

    def test_adjacency_matrix(self, mixed_pdag: PDAG):
        """Test adjacency matrix.

        Args:
            mixed_pdag (PDAG): _description_
        """
        amat = mixed_pdag.adjacency_matrix
        assert amat.shape[0] == amat.shape[1] == mixed_pdag.num_nodes
        assert amat.sum().sum() == mixed_pdag.num_dir_edges + 2 * mixed_pdag.num_undir_edges

    def test_dag2cpdag(self):
        """Test dag to cpdag."""
        edges = [("1", "2"), ("2", "3"), ("3", "4")]
        dag1 = nx.DiGraph(edges)
        cpdag1 = dag2cpdag(dag=dag1)
        assert cpdag1.num_dir_edges == 0
        assert cpdag1.num_undir_edges == len(edges)

        dag2 = nx.DiGraph([("1", "3"), ("2", "3")])
        cpdag2 = dag2cpdag(dag=dag2)
        assert set(cpdag2.dir_edges) == set(dag2.edges)
        assert cpdag2.num_undir_edges == 0

        dag3 = nx.DiGraph([("1", "3"), ("2", "3"), ("1", "4")])
        cpdag3 = dag2cpdag(dag=dag3)
        num_dir_edges = 2
        assert cpdag3.num_dir_edges == num_dir_edges
        assert cpdag3.num_undir_edges == 1

    def test_example_a_to_alldags(self):
        """Test to all DAGs."""
        # Set up CPDAG: a - c - b -> MEC has 3 Members
        pdag = nx.Graph([("a", "c"), ("b", "c")])
        amat = nx.to_pandas_adjacency(pdag)

        # Inititiate
        example_pdag = PDAG.from_pandas_adjacency(pd_amat=amat)

        # Act
        all_dags = example_pdag.to_alldags()
        number_of_dags = 3

        assert len(all_dags) == number_of_dags
        assert all([isinstance(dag, DAG) for dag in all_dags])

    def test_example_b_to_alldags(self):
        """Test to all DAGs."""
        # Set up CPDAG: a - (b,c,d) -> MEC has 4 Members
        pdag = nx.Graph([("a", "b"), ("a", "c"), ("a", "d")])
        amat = nx.to_pandas_adjacency(pdag)

        # Inititiate
        example_pdag = PDAG.from_pandas_adjacency(pd_amat=amat)

        # Act
        all_dags = example_pdag.to_alldags()
        number_of_dags = 4
        assert len(all_dags) == number_of_dags
        assert all([isinstance(dag, DAG) for dag in all_dags])

    def test_empty_graph_to_alldags(self):
        """Test empty graph to all DAGs."""
        # Set up empty PDAG, has exaclty one DAG that is the same as the PDAG.
        pdag = nx.Graph()
        pdag.add_nodes_from(["a", "b", "c", "d"])
        amat = nx.to_pandas_adjacency(pdag)

        # Inititiate
        empty_pdag = PDAG.from_pandas_adjacency(pd_amat=amat)

        # Act
        all_dags = empty_pdag.to_alldags()

        assert len(all_dags) == 1
        assert all_dags[0].edges == list(pdag.edges)
        assert set(all_dags[0].nodes) == set(pdag.nodes)

    def test_to_random_dag(self):
        """Test transform to random DAG."""
        # Set up CPDAG: a - (b,c,d) -> MEC has 4 Members
        pdag = nx.Graph([("a", "b"), ("a", "c"), ("a", "d")])
        amat = nx.to_pandas_adjacency(pdag)

        # Inititiate
        example_pdag = PDAG.from_pandas_adjacency(pd_amat=amat)

        # Act
        random_dag = example_pdag.to_random_dag()

        assert isinstance(random_dag, DAG)
        assert nx.is_directed_acyclic_graph(random_dag.to_networkx())


class TestDAG:
    """Test class for DAGs."""

    @pytest.fixture(scope="class")
    def example_dag(self) -> DAG:
        """Example DAG.

        Returns:
            DAG: _description_
        """
        return DAG(nodes=["A", "B", "C"], edges=[("A", "B"), ("A", "C")])

    def test_instance_is_created(self):
        """Test whether instance is created."""
        dag = DAG(nodes=["A", "B", "C"])
        assert isinstance(dag, DAG)
        assert isinstance(dag, GRAPH)

    def test_edges(self, example_dag: DAG):
        """Test edges added.

        Args:
            example_dag (DAG): _description_
        """
        num_edges = 2
        assert example_dag.num_edges == num_edges
        assert set(example_dag.edges) == {("A", "B"), ("A", "C")}

    def test_children(self, example_dag: DAG):
        """Test children of DAG.

        Args:
            example_dag (DAG): _description_
        """
        assert set(example_dag.children(of_node="A")) == {"B", "C"}
        assert example_dag.children(of_node="B") == []
        assert example_dag.children(of_node="C") == []

    def test_parents(self, example_dag: DAG):
        """Test parents of DAG.

        Args:
            example_dag (DAG): _description_
        """
        assert example_dag.parents(of_node="A") == []
        assert set(example_dag.parents(of_node="B")) == {"A"}
        assert set(example_dag.parents(of_node="C")) == {"A"}

    def test_from_pandas_adjacency(self, example_dag: DAG):
        """Test from pandas works.

        Args:
            example_dag (DAG): _description_
        """
        amat = pd.DataFrame(
            [[0, 1, 1], [0, 0, 0], [0, 0, 0]],
            columns=["A", "B", "C"],
            index=["A", "B", "C"],
        )
        from_pandas_pdag = DAG.from_pandas_adjacency(pd_amat=amat)
        assert np.allclose(from_pandas_pdag.adjacency_matrix, amat)
        assert set(from_pandas_pdag.edges) == set(example_dag.edges)

    def test_remove_edge(self, example_dag: DAG):
        """Test edge removal.

        Args:
            example_dag (DAG): _description_
        """
        assert ("A", "C") in example_dag.edges
        example_dag.remove_edge("A", "C")
        assert ("A", "C") not in example_dag.edges
        with pytest.raises(AssertionError, match=r"Edge does not exist in current DAG"):
            example_dag.remove_edge("B", "A")

    def test_remove_node(self, example_dag: DAG):
        """Test node removal.

        Args:
            example_dag (DAG): _description_
        """
        assert "C" in example_dag.nodes
        example_dag.remove_node("C")
        assert "C" not in example_dag.nodes

    def test_to_cpdag(self):
        """Test to CPDAG works."""
        dag = DAG()
        edges = [("A", "B"), ("A", "C")]
        dag.add_edges_from(edges)
        cpdag = dag.to_cpdag()
        assert isinstance(cpdag, PDAG)
        assert cpdag.num_undir_edges == len(edges)
        assert cpdag.num_dir_edges == 0

    def test_adjacency_matrix(self, example_dag: DAG):
        """Test adjacency matrix property works.

        Args:
            example_dag (DAG): _description_
        """
        amat = example_dag.adjacency_matrix
        assert amat.shape[0] == amat.shape[1] == example_dag.num_nodes
        assert amat.sum().sum() == example_dag.num_edges

    def test_to_networkx(self, example_dag: DAG):
        """Test convert to networkx.

        Args:
            example_dag (DAG): _description_
        """
        nxg = example_dag.to_networkx()
        assert isinstance(nxg, nx.DiGraph)
        assert set(nxg.edges) == set(example_dag.edges)

    def test_from_networkx(self, example_dag: DAG):
        """Test import from networkx.

        Args:
            example_dag (DAG): _description_
        """
        nxg = example_dag.to_networkx()
        from_networkx_dag = DAG.from_nx(nxg)
        assert set(from_networkx_dag.edges) == set(example_dag.edges)

    def test_error_when_cyclic(self):
        """Test error thrown when cyclic."""
        dag = DAG()
        with pytest.raises(ValueError):
            dag.add_edges_from([("A", "C"), ("C", "D"), ("D", "A")])


class TestLayeredDAG:
    """Test LayeredDAG properties."""

    @pytest.fixture(scope="class")
    def example_layered_dag(self) -> LayeredDAG:
        """Example DAG.

        Returns:
            DAG: _description_
        """
        layering = {"L_1": ["A", "B"], "L_2": ["C"]}
        return LayeredDAG(nodes=["A", "B", "C"], edges=[("A", "B"), ("A", "C")], layering=layering)

    def test_instance_is_created(self):
        """Test whether instance is created."""
        ldag = LayeredDAG(nodes=["A", "B", "C"])
        assert isinstance(ldag, LayeredDAG)
        assert isinstance(ldag, DAG)
        assert isinstance(ldag, GRAPH)

    def test_adding_edges_works(self):
        """Test whether adding edges works."""
        ldag = LayeredDAG()
        layering = {"L_1": ["A", "B"], "L_2": ["C"]}
        ldag.layering = layering
        ldag.add_edge(("A", "B"))
        ldag.add_edge(("B", "C"))

        assert set(ldag.edges) == {("A", "B"), ("B", "C")}

    def test_error_if_layering_not_provided(self):
        """Test whether adding edges works."""
        ldag = LayeredDAG()
        with pytest.raises(Exception) as exc_info:
            ldag.add_edge(("A", "B"))

        assert str(exc_info.value) == "Layering must be provided before adding edges."

    def test_error_if_layering_not_respected(self):
        """Test whether adding edges works."""
        ldag = LayeredDAG()
        layering = {"L_1": ["A", "B"], "L_2": ["C"]}
        ldag.layering = layering
        ldag.add_edge(("A", "B"))
        with pytest.raises(Exception) as exc_info:
            ldag.add_edge(("C", "B"))

        assert (
            str(exc_info.value)
            == "The edge set you provided \
                does not agree with the layering.\
                Check your input!"
        )

    def test_layer_induced_subgraph(self, example_layered_dag: LayeredDAG):
        """Test layer induced subgraph."""
        ldag = example_layered_dag
        l_subdag = ldag.layer_induced_subgraph(nodes=["A", "B"])
        assert isinstance(l_subdag, DAG)
        assert set(l_subdag.nodes) == {"A", "B"}
        assert set(l_subdag.edges) == {("A", "B")}

    def test_layer_induced_subgraph_throws_error(self, example_layered_dag: LayeredDAG):
        """Test layer induced subgraph."""
        ldag = example_layered_dag
        with pytest.raises(Exception) as exc_info:
            _ = ldag.layer_induced_subgraph(nodes=["B", "C"])

        assert str(exc_info.value) == "Nodes you provide must correspond to a layer."

    def test_from_pandas_adjacency(self, example_layered_dag: LayeredDAG):
        """Test from_pandas_adjacency works.

        Args:
            example_layered_dag (LayeredDAG): _description_
        """
        ldag = example_layered_dag
        layering = ldag.layering
        amat = pd.DataFrame(
            [[0, 1, 1], [0, 0, 0], [0, 0, 0]],
            columns=["A", "B", "C"],
            index=["A", "B", "C"],
        )
        from_pandas_ldag = LayeredDAG.from_pandas_adjacency(pd_amat=amat, layering=layering)
        assert np.allclose(from_pandas_ldag.adjacency_matrix, amat)
        assert set(from_pandas_ldag.edges) == set(ldag.edges)
        assert from_pandas_ldag.num_edges == ldag.num_edges

    def test_from_networkx(self, example_layered_dag: LayeredDAG):
        """Test import from networkx.

        Args:
            example_layered_dag (DAG): _description_
        """
        layering = example_layered_dag.layering
        nxg = example_layered_dag.to_networkx()
        from_networkx_dag = LayeredDAG.from_nx(nxg, layering)
        assert set(from_networkx_dag.edges) == set(example_layered_dag.edges)


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
