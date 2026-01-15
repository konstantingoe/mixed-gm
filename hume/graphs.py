"""Basic graphical strudtures."""

import logging

logger = logging.getLogger(__name__)


class Graph:
    """Inits the Graph class."""

    def __init__(self, nodes: list[str | int], edges: list[tuple[str | int, str | int]]) -> None:
        """Init.

        Args:
            nodes (list[str  |  int]): Nodes in the graph
            edges (list[tuple[str  |  int, str  |  int]]): Edges in the graph
        """
        self._nodes = nodes
        self._edges = edges

    def __repr__(self) -> str:
        """Returns a string representation of the Graph object.

        Returns:
            str: A string representation of the Graph object.
        """
        return f"Graph(nodes={self.nodes}, edges={self.edges})"

    def __str__(self) -> str:
        """Returns a string representation of the Graph object.

        Returns:
            str: A string representation of the Graph object.
        """
        return f"Graph(nodes={self.nodes}, edges={self.edges})"

    @property
    def n_nodes(self) -> int:
        """Returns the number of nodes in the graph.

        Returns:
            int: The number of nodes in the graph
        """
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        """Returns the number of edges in the graph.

        Returns:
            int: The number of edges in the graph
        """
        return len(self.edges)

    @property
    def nodes(self) -> list[str | int]:
        """Nodes in the graph.

        Returns:
            ist[str | int]: Nodes.
        """
        return self._nodes

    @property
    def edges(self) -> list[tuple[str | int, str | int]]:
        """Edges on graph.

        Returns:
            list[tuple[str | int, str | int]]: Edges.
        """
        return self._edges
