"""
tests/test_attachment_point_feature.py
======================================
Unit tests for the opt-in ``AtomFeature.IsAttachmentPoint`` node feature.

When the feature is in ``node_feats``, PSMILES wildcard atoms (``*``,
atomic number 0) are removed from the graph and the boolean
``IsAttachmentPoint`` column is set to ``1`` on every non-wildcard atom
that was bonded to one. When the feature is absent the graph is built
exactly as before (no stripping, no extra column).

These tests bypass PyG's filesystem processing path — they directly
exercise ``_strip_wildcards``, ``_atom_features`` and
``_build_polymer_graph`` on a half-initialised instance — so they run in
milliseconds with no I/O.
"""

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402

from polynet.config.enums import AtomFeature  # noqa: E402
from polynet.featurizer.polymer_graph import CustomPolymerGraph  # noqa: E402


def _bare_instance(node_feats, *, smiles_col=("smiles",), weights_col=None):
    """
    Build a ``CustomPolymerGraph`` instance without invoking PyG's
    ``Dataset.__init__`` (which would require a CSV on disk). We need only
    the attributes ``_build_polymer_graph`` / ``_atom_features`` read.
    """
    inst = object.__new__(CustomPolymerGraph)
    inst.node_feats = node_feats
    inst.edge_feats = {}
    inst.smiles_col = list(smiles_col)
    inst.weights_col = weights_col
    inst.target_col = None
    inst.id_col = None
    inst.polymer_descriptors = []
    return inst


# ---------------------------------------------------------------------------
# _strip_wildcards: pure helper, no instance needed (it's a @staticmethod)
# ---------------------------------------------------------------------------


def test_strip_wildcards_on_psmiles():
    mol = Chem.MolFromSmiles("[*]CCC[*]")  # 5 atoms: *-C-C-C-*
    stripped, attachment = CustomPolymerGraph._strip_wildcards(mol)
    assert stripped.GetNumAtoms() == 3
    # Surviving atoms = the three carbons (orig 1,2,3) → new 0,1,2.
    # The two end carbons (orig 1 and orig 3) were *-neighbours; the
    # middle carbon (orig 2) was not.
    assert attachment == {0, 2}


def test_strip_wildcards_on_small_molecule():
    mol = Chem.MolFromSmiles("CCO")
    stripped, attachment = CustomPolymerGraph._strip_wildcards(mol)
    assert stripped.GetNumAtoms() == 3  # unchanged
    assert attachment == set()


# ---------------------------------------------------------------------------
# _build_polymer_graph: full feature-vector path
# ---------------------------------------------------------------------------


def _row(smiles, smiles_col="smiles"):
    return pd.Series({smiles_col: smiles}), pd.DataFrame([{smiles_col: smiles}])


def test_psmiles_with_attachment_flag_enabled():
    """3 carbons remain; column 0 = the attachment-point flag."""
    inst = _bare_instance({AtomFeature.IsAttachmentPoint: {}})
    row, df = _row("[*]CCC[*]")
    data = inst._build_polymer_graph(row, df, index=0)

    assert data.x.shape == (3, 1)  # 3 atoms, 1 feature column (IsAttachmentPoint)
    assert data.x.view(-1).tolist() == [1.0, 0.0, 1.0]
    # 2 undirected C-C bonds -> 4 directed entries in edge_index
    assert data.edge_index.shape[1] == 4


def test_psmiles_without_flag_keeps_wildcards():
    """Without the flag in node_feats, behaviour matches the old code: 5 atoms."""
    inst = _bare_instance({AtomFeature.GetIsAromatic: {}})  # any other feature
    row, df = _row("[*]CCC[*]")
    data = inst._build_polymer_graph(row, df, index=0)

    assert data.x.shape == (5, 1)  # 3 C + 2 * = 5 atoms
    # GetIsAromatic for all 5 atoms is False (0)
    assert data.x.view(-1).tolist() == [0.0] * 5
    assert data.edge_index.shape[1] == 8  # 4 undirected bonds × 2


def test_small_molecule_with_flag_enabled_yields_all_zero_column():
    """Non-PSMILES inputs degrade gracefully: same atom count, flag is all 0."""
    inst = _bare_instance({AtomFeature.IsAttachmentPoint: {}})
    row, df = _row("CCO")
    data = inst._build_polymer_graph(row, df, index=0)

    assert data.x.shape == (3, 1)
    assert data.x.view(-1).tolist() == [0.0, 0.0, 0.0]


def test_dimensionality_consistent_across_psmiles_and_small_molecule():
    """
    Same node_feats config must produce graphs with the same feature-vector
    dimensionality for PSMILES and small molecules — so they can sit in one
    batch without padding.
    """
    inst = _bare_instance({AtomFeature.IsAttachmentPoint: {}})
    row_p, df_p = _row("[*]CCC[*]")
    row_s, df_s = _row("CCO")
    d_p = inst._build_polymer_graph(row_p, df_p, index=0)
    d_s = inst._build_polymer_graph(row_s, df_s, index=1)

    assert d_p.x.shape[1] == d_s.x.shape[1]
