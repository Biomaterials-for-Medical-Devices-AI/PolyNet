"""
tests/test_model_backcompat.py
==============================
Back-compatibility guarantees for GNN model persistence after the
cross-attention removal.

GNN models are persisted as whole pickled module objects
(``torch.save(model)``), not state dicts. Removing the unused ``cross_att``
constructor argument and the ``_cross_attention`` / ``monomer_W_att`` members
must not break:

1. a normal save → load → predict round-trip, and
2. loading an *old* instance that still carries a stale ``cross_att`` attribute
   (pickle restores instance attributes without calling ``__init__``, and the
   forward pass no longer references the attribute).
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from polynet.models.gnn.gcn import GCNRegressor  # noqa: E402
from polynet.models.persistence import load_gnn_model, save_gnn_model  # noqa: E402


def _tiny_inputs():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    edge_attr = torch.randn(4, 3)
    batch = torch.zeros(4, dtype=torch.long)
    return x, edge_index, edge_attr, batch


def test_save_load_predict_roundtrip(tmp_path):
    model = GCNRegressor(improved=True, n_node_features=8, n_edge_features=3, seed=0)
    model.eval()
    x, ei, ea, b = _tiny_inputs()

    before = model.predict(x=x, edge_index=ei, edge_attr=ea, batch_index=b)

    path = tmp_path / "model.pt"
    save_gnn_model(model, path)
    reloaded = load_gnn_model(path)
    reloaded.eval()

    after = reloaded.predict(x=x, edge_index=ei, edge_attr=ea, batch_index=b)
    assert after == pytest.approx(before, rel=1e-6)


def test_legacy_cross_att_attribute_is_tolerated(tmp_path):
    """A model carrying a stale ``cross_att`` attribute (as an old pickle would)
    still loads and predicts — the attribute is simply ignored."""
    model = GCNRegressor(improved=True, n_node_features=8, n_edge_features=3, seed=0)
    # Simulate the attribute set by the pre-migration ``__init__``.
    model.cross_att = False
    model.eval()

    path = tmp_path / "legacy_model.pt"
    save_gnn_model(model, path)
    reloaded = load_gnn_model(path)
    reloaded.eval()

    x, ei, ea, b = _tiny_inputs()
    out = reloaded.predict(x=x, edge_index=ei, edge_attr=ea, batch_index=b)
    assert out.shape == (1,)
