"""
tests/test_chem_utils.py
=========================
Unit tests for ``polynet.utils.chem_utils``.

These tests invoke RDKit but require no GPU, network access, or trained models.

Function under test: ``polynet.utils.chem_utils.fragment_and_match``
"""

import pytest

from polynet.config.enums import FragmentationMethod
from polynet.utils.chem_utils import fragment_and_match


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def peg_acrylate_smiles():
    """PEG-acrylate monomer — long chain with repeating ether units."""
    return "C=CC(=O)OCCOCCOCCOCCOCCOCCOCCO"


@pytest.fixture
def peg_acrylate_expected():
    return {
        "C=CC=O": [[0, 1, 2, 3]],
        "O": [[4], [7], [10], [13], [16], [19], [22]],
        "CC": [[5, 6], [8, 9], [11, 12], [14, 15], [17, 18], [20, 21]],
        "CCO": [[23, 24, 25]],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFragmentAndMatchBRICS:
    def test_peg_acrylate_fragment_keys(self, peg_acrylate_smiles, peg_acrylate_expected):
        """Fragment SMILES keys match the expected set."""
        result = fragment_and_match(peg_acrylate_smiles, FragmentationMethod.BRICS)
        assert set(result.keys()) == set(peg_acrylate_expected.keys())

    def test_peg_acrylate_atom_indices(self, peg_acrylate_smiles, peg_acrylate_expected):
        """Atom index lists for each fragment match exactly."""
        result = fragment_and_match(peg_acrylate_smiles, FragmentationMethod.BRICS)
        for frag_smiles, expected_idx_lists in peg_acrylate_expected.items():
            assert result[frag_smiles] == expected_idx_lists, (
                f"Index mismatch for fragment '{frag_smiles}': "
                f"got {result[frag_smiles]}, expected {expected_idx_lists}"
            )

    def test_peg_acrylate_all_atoms_covered(self, peg_acrylate_smiles):
        """Every atom in the molecule appears in exactly one fragment."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles(peg_acrylate_smiles)
        n_atoms = mol.GetNumAtoms()

        result = fragment_and_match(peg_acrylate_smiles, FragmentationMethod.BRICS)
        covered = sorted(i for idx_lists in result.values() for idxs in idx_lists for i in idxs)
        assert covered == list(range(n_atoms)), "Not all atoms are covered without overlap"

    def test_no_brics_bonds_returns_whole_molecule(self):
        """A molecule with no BRICS-breakable bonds is returned as a single fragment."""
        smiles = "CCO"  # ethanol — no BRICS bonds
        result = fragment_and_match(smiles, FragmentationMethod.BRICS)
        assert len(result) == 1
        idx_lists = next(iter(result.values()))
        assert idx_lists == [[0, 1, 2]]

    def test_returns_nonempty_dict_for_any_valid_mol(self):
        """Result is always a non-empty dict for any valid SMILES."""
        cases = ["C", "CCO", "c1ccccc1", "CC(=O)O", "C=CC(=O)OCCOCCO"]
        for smi in cases:
            result = fragment_and_match(smi, FragmentationMethod.BRICS)
            assert isinstance(result, dict) and len(result) > 0, (
                f"Empty result for SMILES '{smi}'"
            )

    def test_invalid_smiles_raises(self):
        """An invalid SMILES string raises a ValueError."""
        with pytest.raises(ValueError, match="Invalid SMILES"):
            fragment_and_match("not_a_smiles", FragmentationMethod.BRICS)
