"""Module for dealing with graphs."""

from __future__ import annotations

import logging
from abc import (
    ABCMeta,
    abstractmethod,
)
from collections import defaultdict
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GRAPH(metaclass=ABCMeta):
    """Abstract base class for all Graphs in current project."""

    def __init__(self) -> None:
        """Init ABC."""
        pass

    @property
    @abstractmethod
    def adjacency_matrix(self) -> pd.DataFrame:
        """Return adjacency matrix.

        Raises:
            AssertionError: _description_
            AssertionError: _description_
            ValueError: _description_
            AssertionError: _description_
            AssertionError: _description_
            TypeError: _description_

        Returns:
            pd.DataFrame: Adjacency matrix of underlying graph.
        """

    @property
    @abstractmethod
    def causal_order(self) -> list[str] | None:
        """Return causal order.

        Raises:
            AssertionError: _description_
            AssertionError: _description_
            ValueError: _description_
            AssertionError: _description_
            AssertionError: _description_
            TypeError: _description_

        Returns:
            list[str] | None: Causal order of underlying graph.
                None if not a DAG.
        """


class UGRAPH(GRAPH):
    """Class for dealing with undirected graph i.e. graphs that only contain undirected edges."""

    def __init__(
        self,
        nodes: list[str] | None = None,
        edges: list[tuple[str, str]] | None = None,
    ) -> None:
        """UGRAPH constructor.

        Args:
            nodes (list[str] | None, optional): Nodes. Defaults to None.
            edges (list[tuple[str,str]] | None, optional): Edges. Defaults to None.
        """
        if nodes is None:
            nodes = []
        if edges is None:
            edges = []

        self._nodes: set[str] = set(nodes)
        self._edges: set[tuple[str, str]] = set()
        self._neighbors: defaultdict[str, set[str]] = defaultdict(set)

        for edge in edges:
            self._add_edge(*edge)

    def _add_edge(self, i: str, j: str) -> None:
        self._nodes.add(i)
        self._nodes.add(j)
        self._edges.add((i, j))

        self._neighbors[i].add(j)
        self._neighbors[j].add(i)

    def neighbors(self, node: str) -> set[str]:
        """Gives all neighbors of node `node`.

        Args:
            node (str): node in current UGRAPH.

        Returns:
            set: set of neighbors.
        """
        if node in self._neighbors:
            return self._neighbors[node]
        else:
            return set()

    def is_adjacent(self, i: str, j: str) -> bool:
        """Return True if the graph contains an undirected edge between i and j.

        Args:
            i (str): node i.
            j (str): node j.

        Returns:
            bool: True if i - j
        """
        return (i, j) in self._edges or (j, i) in self._edges

    def is_clique(self, potential_clique: set[str]) -> bool:
        """Check every pair of nodes in potential_clique is adjacent."""
        return all(self.is_adjacent(i, j) for i, j in combinations(potential_clique, 2))

    @classmethod
    def from_pandas_adjacency(cls, pd_amat: pd.DataFrame) -> UGRAPH:
        """Build UGRAPH from a Pandas adjacency matrix.

        Args:
            pd_amat (pd.DataFrame): input adjacency matrix.

        Returns:
            UGRAPH
        """
        assert pd_amat.shape[0] == pd_amat.shape[1]
        nodes = list(pd_amat.columns)

        all_connections = []
        start, end = np.where(pd_amat != 0)
        for idx, _ in enumerate(start):
            all_connections.append((pd_amat.columns[start[idx]], pd_amat.columns[end[idx]]))

        edges = [tuple(item) for item in set(frozenset(item) for item in all_connections)]

        return UGRAPH(nodes=nodes, edges=edges)

    def remove_edge(self, i: str, j: str) -> None:
        """Removes edge in question.

        Args:
            i (str): first node
            j (str): second node

        Raises:
            AssertionError: if edge does not exist
        """
        if not self.is_adjacent(i, j):
            raise AssertionError("Edge does not exist in current UGRAPH")

        self._edges.discard((i, j))
        self._edges.discard((j, i))
        self._neighbors[i].discard(j)
        self._neighbors[j].discard(i)

    def remove_node(self, node: str) -> None:
        """Remove a node from the graph.

        Args:
            node (str): node to remove
        """
        self._nodes.remove(node)

        self._edges = {(i, j) for i, j in self._edges if node not in {i, j}}

        for nbr in self._neighbors[node]:
            self._neighbors[nbr].discard(node)

        self._neighbors.pop(node, "I was never here")

    @property
    def adjacency_matrix(self) -> pd.DataFrame:
        """Returns adjacency matrix.

        The i,jth entry being one indicates that there is an undirected edge
        between i and j. A zero indicates that there is no edge. The matrix
        is symmetric.

        Returns:
            pd.DataFrame: adjacency matrix
        """
        amat = pd.DataFrame(
            np.zeros([self.num_nodes, self.num_nodes]),
            index=self.nodes,
            columns=self.nodes,
        )
        for edge in self.edges:
            amat.loc[edge] = amat.loc[edge[::-1]] = 1
        return amat

    @property
    def causal_order(self) -> None:
        """Causal order is None.

        This is because undirected graphs do not imply a causal order.

        Returns:
            None: None
        """
        return None

    def copy(self) -> UGRAPH:
        """Return a copy of the graph."""
        return UGRAPH(nodes=list(self._nodes), edges=list(self._edges))

    def show(self) -> None:
        """Plot UGRAPH."""
        graph = self.to_networkx()
        pos = nx.circular_layout(graph)
        nx.draw(graph, pos=pos, with_labels=True)

    def to_networkx(self) -> nx.Graph:
        """Convert to networkx graph.

        Returns:
            nx.Graph: Undirected networkx graph.
        """
        nx_ugraph = nx.Graph()
        nx_ugraph.add_nodes_from(self.nodes)
        nx_ugraph.add_edges_from(self.edges)
        return nx_ugraph

    @property
    def nodes(self) -> list[str]:
        """Get all nodes in current UGRAPH.

        Returns:
            list: list of nodes.
        """
        return sorted(list(self._nodes))

    @property
    def num_nodes(self) -> int:
        """Number of nodes in current UGRAPH.

        Returns:
            int: Number of nodes
        """
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        """Number of edges in current UGRAPH.

        Returns:
            int: Number of edges
        """
        return len(self._edges)

    @property
    def edges(self) -> list[tuple[str, str]]:
        """Gives all edges in current UGRAPH.

        Returns:
            list[tuple[str,str]]: List of edges.
        """
        return list(self._edges)
