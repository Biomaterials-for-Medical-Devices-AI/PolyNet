"""
tests/test_ratio_normalization.py
=================================
Unit tests for per-polymer molar-ratio normalisation in the graph featuriser.

``CustomPolymerGraph._build_polymer_graph`` stores a per-node ``weight_monomer``
equal to ``ratio_m / Σ ratios`` over the participating monomers, so each
polymer's weights sum to 1 regardless of the input scale. When the ratios
already sum to 100 this reproduces the previous ``ratio / 100`` behaviour
exactly.

These tests bypass PyG's filesystem path (see ``test_attachment_point_feature``)
by building a half-initialised instance, so they run in milliseconds.
"""

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")

from polynet.config.enums import AtomFeature  # noqa: E402
from polynet.featurizer.polymer_graph import CustomPolymerGraph  # noqa: E402


def _bare_instance(smiles_cols, weights_col):
    inst = object.__new__(CustomPolymerGraph)
    inst.node_feats = {AtomFeature.GetIsAromatic: {}}  # 1 feature column per atom
    inst.edge_feats = {}
    inst.smiles_col = list(smiles_cols)
    inst.weights_col = weights_col
    inst.target_col = None
    inst.id_col = None
    inst.polymer_descriptors = []
    return inst


def _per_monomer_weights(data) -> list[float]:
    """Return the (constant) ``weight_monomer`` value for each monomer, in order."""
    wid = data.monomer_id.view(-1)
    w = data.weight_monomer.view(-1)
    return [float(w[wid == m][0]) for m in sorted(set(wid.tolist()))]


def _build(smiles_cols, weights_col, values):
    inst = _bare_instance(smiles_cols, weights_col)
    row = pd.Series(values)
    df = pd.DataFrame([values])
    return inst._build_polymer_graph(row, df, index=0)


def test_percentage_ratios_match_old_divide_by_100():
    """60/40 percentages → weights 0.6/0.4 (identical to the previous behaviour)."""
    data = _build(
        ["m1", "m2"], {"m1": "r1", "m2": "r2"}, {"m1": "C", "m2": "CC", "r1": 60.0, "r2": 40.0}
    )
    assert _per_monomer_weights(data) == pytest.approx([0.6, 0.4])


def test_fraction_ratios_normalise_the_same():
    """0.6/0.4 fractions normalise to the same 0.6/0.4 as the percentage form."""
    data = _build(
        ["m1", "m2"], {"m1": "r1", "m2": "r2"}, {"m1": "C", "m2": "CC", "r1": 0.6, "r2": 0.4}
    )
    assert _per_monomer_weights(data) == pytest.approx([0.6, 0.4])


def test_three_monomers_ratio_1_1_2():
    """Ratios 1:1:2 across three monomers normalise to 0.25/0.25/0.5."""
    data = _build(
        ["m1", "m2", "m3"],
        {"m1": "r1", "m2": "r2", "m3": "r3"},
        {"m1": "C", "m2": "CC", "m3": "O", "r1": 1.0, "r2": 1.0, "r3": 2.0},
    )
    assert _per_monomer_weights(data) == pytest.approx([0.25, 0.25, 0.5])


def test_weights_sum_to_one():
    """The per-node weights of any polymer sum to 1 across participating monomers."""
    data = _build(
        ["m1", "m2", "m3"],
        {"m1": "r1", "m2": "r2", "m3": "r3"},
        {"m1": "C", "m2": "CC", "m3": "CCC", "r1": 3.0, "r2": 5.0, "r3": 7.0},
    )
    assert sum(_per_monomer_weights(data)) == pytest.approx(1.0)


def test_zero_ratio_monomer_excluded_and_homopolymer_weight_is_one():
    """A 100/0 row keeps only monomer A, whose normalised weight is 1.0."""
    data = _build(
        ["m1", "m2"], {"m1": "r1", "m2": "r2"}, {"m1": "C", "m2": "CC", "r1": 100.0, "r2": 0.0}
    )
    # Only the first monomer (1 atom) participates.
    assert len(data.mols) == 1
    assert _per_monomer_weights(data) == pytest.approx([1.0])
