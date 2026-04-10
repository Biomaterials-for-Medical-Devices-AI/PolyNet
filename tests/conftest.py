"""
tests/conftest.py
=================
Shared pytest fixtures for the PolyNet test suite.

All fixtures here are available to every test module without importing.
"""

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# SMILES / polymer DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture
def two_monomer_df() -> pd.DataFrame:
    """
    Two-monomer polymer dataset with weight fractions.

    Weights are in **percentage** scale (0–100) to match the
    ``merge_weighted`` formula which divides by 100 internally.

    Molecules chosen for simple, verifiable RDKit descriptor values:
    - smiles_A: methane (C), ethane (CC), propane (CCC)
    - smiles_B: water (O), methanol (CO), ethanol (CCO)
    """
    return pd.DataFrame(
        {
            "smiles_A": ["C", "CC", "CCC"],
            "smiles_B": ["O", "CO", "CCO"],
            "weight_A": [60.0, 40.0, 50.0],  # percentage scale
            "weight_B": [40.0, 60.0, 50.0],
            "Tg": [300.0, 350.0, 375.0],
        }
    )


@pytest.fixture
def single_monomer_df() -> pd.DataFrame:
    """Single-monomer dataset for ``NoMerging`` strategy tests."""
    return pd.DataFrame(
        {
            "smiles_A": ["C", "CC", "CCC"],
            "Tg": [300.0, 350.0, 375.0],
        }
    )


# ---------------------------------------------------------------------------
# df_dict fixtures (manually constructed — no RDKit needed)
# ---------------------------------------------------------------------------


@pytest.fixture
def known_df_dict() -> dict[str, pd.DataFrame]:
    """
    Two-key df_dict with integer-friendly values for exact numeric assertions.

    smiles_A row values: MolWt=[16, 30, 44], HeavyAtomCount=[1, 2, 3]
    smiles_B row values: MolWt=[18, 32, 46], HeavyAtomCount=[1, 2, 3]

    Expected averages:  MolWt=[17, 31, 45], HeavyAtomCount=[1, 2, 3]
    """
    idx = [0, 1, 2]
    df_a = pd.DataFrame({"MolWt": [16.0, 30.0, 44.0], "HeavyAtomCount": [1.0, 2.0, 3.0]}, index=idx)
    df_b = pd.DataFrame({"MolWt": [18.0, 32.0, 46.0], "HeavyAtomCount": [1.0, 2.0, 3.0]}, index=idx)
    return {"smiles_A": df_a, "smiles_B": df_b}


@pytest.fixture
def single_key_df_dict() -> dict[str, pd.DataFrame]:
    """Single-key df_dict for ``NoMerging`` / ``_single_smiles`` tests."""
    idx = [0, 1, 2]
    df = pd.DataFrame({"MolWt": [16.0, 30.0, 44.0], "HeavyAtomCount": [1.0, 2.0, 3.0]}, index=idx)
    return {"smiles_A": df}


@pytest.fixture
def empty_data_index() -> pd.DataFrame:
    """
    Empty-column DataFrame that mimics the ``data[[]]`` pattern used by
    ``compute_rdkit_descriptors`` so no extra index columns appear in output.
    """
    return pd.DataFrame(index=[0, 1, 2])


# ---------------------------------------------------------------------------
# Descriptor name lists
# ---------------------------------------------------------------------------

#: Small set of well-known RDKit descriptors — fast and deterministic.
FAST_DESCRIPTORS: list[str] = ["MolWt", "TPSA", "NumRotatableBonds"]
