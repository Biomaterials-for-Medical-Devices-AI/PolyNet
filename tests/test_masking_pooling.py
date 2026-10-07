"""
tests/test_masking_pooling.py
=============================
The masked prediction in substructure masking must go through the same pooling
as the full prediction, so that the attribution Y_full − Y_masked only reflects
the removed nodes and not a difference in how the graph is pooled.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from polynet.config.enums import ApplyWeightingToGraph, Pooling  # noqa: E402
from polynet.explainability.masking import (  # noqa: E402
    MASKING_CACHE_VERSION,
    MASKING_CACHE_VERSION_KEY,
    _predict_from_node_embeddings,
    load_masking_cache,
)
from polynet.models.gnn.gcn import GCNRegressor  # noqa: E402

# Copolymer graph: monomer 0 has 3 nodes (w=0.3), monomer 1 has 5 nodes (w=0.7).
_N_NODES = 8
_EDGE_INDEX = torch.tensor(
    [[0, 1, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7], [1, 0, 2, 1, 4, 3, 5, 4, 6, 5, 7, 6]]
)
_MONOMER_ID = torch.tensor([[0]] * 3 + [[1]] * 5)
_WEIGHT = torch.tensor([[0.3]] * 3 + [[0.7]] * 5)


def _model(weighting, pooling, n_polymer_descriptors=0):
    model = GCNRegressor(
        improved=False,
        n_node_features=4,
        n_edge_features=1,
        pooling=pooling,
        embedding_dim=8,
        apply_weighting_to_graph=weighting,
        n_polymer_descriptors=n_polymer_descriptors,
    )
    model.eval()
    return model


@pytest.mark.parametrize("weighting", list(ApplyWeightingToGraph))
@pytest.mark.parametrize("pooling", list(Pooling))
def test_keeping_every_node_reproduces_the_full_prediction(weighting, pooling):
    torch.manual_seed(0)
    x = torch.rand(_N_NODES, 4)
    descriptors = torch.rand(1, 2)
    model = _model(weighting, pooling, n_polymer_descriptors=2)

    with torch.no_grad():
        full = model.forward(
            x=x,
            edge_index=_EDGE_INDEX,
            monomer_weight=_WEIGHT,
            monomer_id=_MONOMER_ID,
            polymer_descriptors=descriptors,
        )
        h = model.get_node_embeddings(x=x, edge_index=_EDGE_INDEX, monomer_weight=_WEIGHT)
        masked = _predict_from_node_embeddings(
            model,
            h=h,
            keep=torch.ones(_N_NODES, dtype=torch.bool),
            monomer_weight=_WEIGHT,
            monomer_id=_MONOMER_ID,
            polymer_descriptors=descriptors,
        )

    assert torch.allclose(full, masked, atol=1e-6)


def test_masking_a_whole_monomer_drops_it_from_per_monomer_pooling():
    """With per-monomer pooling, removing monomer 0 leaves 0.7 · pool(monomer 1)."""
    torch.manual_seed(0)
    x = torch.rand(_N_NODES, 4)
    model = _model(ApplyWeightingToGraph.PerMonomerPooling, Pooling.GlobalMeanPool)

    with torch.no_grad():
        h = model.get_node_embeddings(x=x, edge_index=_EDGE_INDEX, monomer_weight=_WEIGHT)
        keep = _MONOMER_ID.view(-1) == 1
        masked = _predict_from_node_embeddings(
            model, h, keep, _WEIGHT, _MONOMER_ID, polymer_descriptors=None
        )
        expected = model.readout_function(0.7 * h[keep].mean(dim=0, keepdim=True))

    assert torch.allclose(masked, expected, atol=1e-6)


def test_stale_masking_cache_is_discarded():
    stale = {"GCN": {"1": {"mol": {"chemistry_masking": {}}}}}
    current = {**stale, MASKING_CACHE_VERSION_KEY: MASKING_CACHE_VERSION}

    assert load_masking_cache(stale) == {}
    assert load_masking_cache(current) == current
    assert load_masking_cache({}) == {}
