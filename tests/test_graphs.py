"""Test for hume/graphs.py."""

import pytest

from hume.graphs import Graph


@pytest.mark.parametrize("nodes, edges", [pytest.param(["1", "2", "3"], [("1", "2"), ("2", "3")])])
def test_graphs(nodes: list[str], edges: list[tuple[str]]) -> None:
    """Test graph init.

    Args:
        nodes (_type_): _description_
        edges (_type_): _description_
    """
    graph = Graph(nodes=nodes, edges=edges)
    assert graph.nodes == ["1", "2", "3"]
    assert graph.edges == [("1", "2"), ("2", "3")]
