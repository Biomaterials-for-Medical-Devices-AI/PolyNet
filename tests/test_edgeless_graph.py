"""
tests/test_edgeless_graph.py
============================
Regression tests for bond-free / edgeless graphs through edge-aware GNNs.

A polymer whose graph has no bonds (e.g. only single-atom monomers, or a
PSMILES monomer that reduces to one atom after wildcard stripping) can reach a
convolution with a malformed 1-D empty ``edge_attr`` of shape ``(0,)``. Every
edge-aware architecture must coerce this to ``(0, n_edge_features)`` before its
convolution. The architectures that override ``get_graph_embedding`` /
``get_node_embeddings`` (CGGNN, MPNN) were missing this coercion, which produced
``mat1 and mat2 shapes cannot be multiplied`` inside CGConv / NNConv.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from polynet.models.gnn.cggnn import CGGNNRegressor  # noqa: E402
from polynet.models.gnn.gat import GATRegressor  # noqa: E402
from polynet.models.gnn.mpnn import MPNNRegressor  # noqa: E402
from polynet.models.gnn.transformer import TransformerGNNRegressor  # noqa: E402

_EDGE_AWARE = [
    (CGGNNRegressor, {}),
    (MPNNRegressor, {}),
    (GATRegressor, {"num_heads": 2}),
    (TransformerGNNRegressor, {"num_heads": 2}),
]


def _edgeless_inputs():
    x = torch.randn(3, 8)
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    edge_attr = torch.zeros(0)  # malformed 1-D (0,), as old caches / collation can yield
    batch = torch.zeros(3, dtype=torch.long)
    return x, edge_index, edge_attr, batch


@pytest.mark.parametrize("cls,kw", _EDGE_AWARE)
def test_edgeless_forward(cls, kw):
    model = cls(n_node_features=8, n_edge_features=9, embedding_dim=32, **kw)
    model.eval()
    x, ei, ea, b = _edgeless_inputs()
    out = model.forward(x=x, edge_index=ei, edge_attr=ea, batch_index=b)
    assert out.shape == (1, 1)


@pytest.mark.parametrize("cls,kw", _EDGE_AWARE)
def test_edgeless_node_embeddings(cls, kw):
    """The explainability path calls get_node_embeddings — it must also be safe."""
    model = cls(n_node_features=8, n_edge_features=9, embedding_dim=32, **kw)
    model.eval()
    x, ei, ea, b = _edgeless_inputs()
    h = model.get_node_embeddings(x=x, edge_index=ei, edge_attr=ea, batch_index=b)
    assert h.shape == (3, 32)
