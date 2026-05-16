"""
tests/test_per_monomer_pool.py
==============================
Unit tests for the ``PerMonomerPooling`` weighting strategy.

These exercise ``BaseNetwork._pool`` / ``_pool_per_monomer`` directly with
hand-built node tensors (no message passing, no RDKit), so they run in
milliseconds. ``BaseNetwork.__init__`` only sets attributes and ``pooling_fn``
— it does not require conv layers — so the pooling math can be tested in
isolation.

The key invariant under test (TEST 2) is the one behind the original bug
report: a second monomer present at ratio 0 must not change the polymer
embedding, regardless of its structure.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from torch_geometric.nn import global_mean_pool  # noqa: E402

from polynet.config.enums import ApplyWeightingToGraph, Pooling  # noqa: E402
from polynet.models.base import BaseNetwork  # noqa: E402


def _make_model(pooling: Pooling) -> BaseNetwork:
    model = BaseNetwork(
        n_node_features=2,
        n_edge_features=1,
        pooling=pooling,
        apply_weighting_to_graph=ApplyWeightingToGraph.PerMonomerPooling,
    )
    model.eval()
    return model


# Batch of 2 polymers (7 nodes total), used by several tests:
#   Polymer 0 (homopolymer): monomer A, 2 atoms, w=1.0
#   Polymer 1 (copolymer):   monomer A', 2 atoms, w=0.25
#                            monomer B,  3 atoms, w=0.75
_xA = torch.tensor([[1.0, 2.0], [3.0, 4.0]])              # mean [2, 3]
_xA2 = torch.tensor([[10.0, 10.0], [20.0, 20.0]])         # mean [15, 15]
_xB = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])  # mean [1, 0]
_X = torch.cat([_xA, _xA2, _xB], dim=0)
_BATCH = torch.tensor([0, 0, 1, 1, 1, 1, 1])
_MONOMER_ID = torch.tensor([[0], [0], [0], [0], [1], [1], [1]])
_WEIGHT = torch.tensor(
    [[1.0], [1.0], [0.25], [0.25], [0.75], [0.75], [0.75]]
)


def test_weighted_per_monomer_mean_pool():
    """w1*mean(m1) + w2*mean(m2), per polymer."""
    model = _make_model(Pooling.GlobalMeanPool)
    out = model._pool(_X, _BATCH, _WEIGHT, _MONOMER_ID)

    exp0 = 1.0 * torch.tensor([2.0, 3.0])
    exp1 = 0.25 * torch.tensor([15.0, 15.0]) + 0.75 * torch.tensor([1.0, 0.0])
    expected = torch.stack([exp0, exp1])

    assert torch.allclose(out, expected, atol=1e-5)


def test_zero_weight_monomer_is_invariant():
    """
    A second monomer at ratio 0 must not affect the embedding,
    no matter its structure/features (the original bug).
    """
    model = _make_model(Pooling.GlobalMeanPool)

    junk = torch.tensor([[99.0, -99.0], [7.0, 7.0], [0.0, 5.0], [3.0, 3.0]])
    x = torch.cat([_xA, junk], dim=0)
    batch = torch.tensor([0, 0, 0, 0, 0, 0])
    monomer_id = torch.tensor([[0], [0], [1], [1], [1], [1]])
    weight = torch.tensor([[1.0], [1.0], [0.0], [0.0], [0.0], [0.0]])

    with_phantom = model._pool(x, batch, weight, monomer_id)
    pure = model._pool(
        _xA,
        torch.tensor([0, 0]),
        torch.tensor([[1.0], [1.0]]),
        torch.tensor([[0], [0]]),
    )
    assert torch.allclose(with_phantom, pure, atol=1e-6)


def test_pooling_agnostic_sum():
    """Same blend rule must hold for GlobalAddPool: w1*sum1 + w2*sum2."""
    model = _make_model(Pooling.GlobalAddPool)
    out = model._pool(_X, _BATCH, _WEIGHT, _MONOMER_ID)

    exp0 = 1.0 * _xA.sum(0)
    exp1 = 0.25 * _xA2.sum(0) + 0.75 * _xB.sum(0)
    expected = torch.stack([exp0, exp1])

    assert torch.allclose(out, expected, atol=1e-5)


def test_fallback_when_monomer_id_missing():
    """
    Mode active but monomer_id absent (e.g. no-weights datasets) ->
    fall back to plain pooling over the whole polymer, no crash.
    """
    model = _make_model(Pooling.GlobalMeanPool)
    out = model._pool(_X, _BATCH, _WEIGHT, None)
    plain = global_mean_pool(_X, _BATCH)
    assert torch.allclose(out, plain, atol=1e-5)
