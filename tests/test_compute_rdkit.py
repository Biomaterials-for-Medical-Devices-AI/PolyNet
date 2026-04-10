"""
tests/test_compute_rdkit.py
============================
Integration tests for the ``compute_rdkit_descriptors`` public API.

These tests actually invoke RDKit (~1–2 s total) but do **not** require GPU,
PolyBERT, PolyMetriX, or any network access.

Function under test: ``polynet.featurizer.compute_rdkit_descriptors``
"""

import pandas as pd
import pytest

from polynet.config.enums import DescriptorMergingMethod
from polynet.featurizer import compute_rdkit_descriptors

# Fast RDKit descriptors with known approximate values for simple molecules.
DESCRIPTORS = ["MolWt", "TPSA", "NumRotatableBonds"]

# Approximate MolWt values for the simple test molecules (RDKit average masses).
MOLWT_METHANE = pytest.approx(16.043, abs=0.01)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_monomer_df() -> pd.DataFrame:
    """Two-monomer dataset with percentage-scale weight fractions."""
    return pd.DataFrame(
        {
            "smiles_A": ["C", "CC", "CCC"],
            "smiles_B": ["O", "CO", "CCO"],
            "weight_A": [60.0, 40.0, 50.0],
            "weight_B": [40.0, 60.0, 50.0],
            "Tg": [300.0, 350.0, 375.0],
        }
    )


@pytest.fixture
def single_monomer_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "smiles_A": ["C", "CC", "CCC"],
            "Tg": [300.0, 350.0, 375.0],
        }
    )


@pytest.fixture
def weights_col() -> dict[str, str]:
    return {"smiles_A": "weight_A", "smiles_B": "weight_B"}


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


class TestOutputShape:
    def test_weighted_average_has_one_col_per_descriptor(self, two_monomer_df, weights_col):
        result = compute_rdkit_descriptors(
            data=two_monomer_df,
            smiles_cols=["smiles_A", "smiles_B"],
            descriptor_names=DESCRIPTORS,
            merging_approach=DescriptorMergingMethod.WeightedAverage,
            weights_col=weights_col,
        )
        assert result.shape == (len(two_monomer_df), len(DESCRIPTORS))

    def test_concatenate_has_two_cols_per_descriptor(self, two_monomer_df):
        """Concatenate doubles the column count (one set per SMILES column)."""
        result = compute_rdkit_descriptors(
            data=two_monomer_df,
            smiles_cols=["smiles_A", "smiles_B"],
            descriptor_names=DESCRIPTORS,
            merging_approach=DescriptorMergingMethod.Concatenate,
        )
        assert result.shape == (len(two_monomer_df), len(DESCRIPTORS) * 2)

    def test_row_count_matches_input(self, two_monomer_df):
        result = compute_rdkit_descriptors(
            data=two_monomer_df,
            smiles_cols=["smiles_A", "smiles_B"],
            descriptor_names=DESCRIPTORS,
            merging_approach=DescriptorMergingMethod.Concatenate,
        )
        assert len(result) == len(two_monomer_df)

    def test_single_smiles_no_merging_shape(self, single_monomer_df):
        result = compute_rdkit_descriptors(
            data=single_monomer_df,
            smiles_cols=["smiles_A"],
            descriptor_names=DESCRIPTORS,
            merging_approach=DescriptorMergingMethod.NoMerging,
        )
        assert result.shape == (len(single_monomer_df), len(DESCRIPTORS))


# ---------------------------------------------------------------------------
# Column cleanliness
# ---------------------------------------------------------------------------


class TestColumnCleanliness:
    def test_target_column_not_in_output(self, two_monomer_df):
        result = compute_rdkit_descriptors(
            data=two_monomer_df,
            smiles_cols=["smiles_A", "smiles_B"],
            descriptor_names=DESCRIPTORS,
            merging_approach=DescriptorMergingMethod.Concatenate,
        )
        assert "Tg" not in result.columns

    def test_smiles_columns_not_in_output(self, two_monomer_df):
        result = compute_rdkit_descriptors(
            data=two_monomer_df,
            smiles_cols=["smiles_A", "smiles_B"],
            descriptor_names=DESCRIPTORS,
            merging_approach=DescriptorMergingMethod.Concatenate,
        )
        assert "smiles_A" not in result.columns
        assert "smiles_B" not in result.columns

    def test_weight_columns_not_in_output(self, two_monomer_df, weights_col):
        result = compute_rdkit_descriptors(
            data=two_monomer_df,
            smiles_cols=["smiles_A", "smiles_B"],
            descriptor_names=DESCRIPTORS,
            merging_approach=DescriptorMergingMethod.WeightedAverage,
            weights_col=weights_col,
        )
        assert "weight_A" not in result.columns
        assert "weight_B" not in result.columns

    def test_concatenate_column_names_prefixed(self, two_monomer_df):
        """Concatenated columns are named ``<smiles_col>_<descriptor>``."""
        result = compute_rdkit_descriptors(
            data=two_monomer_df,
            smiles_cols=["smiles_A", "smiles_B"],
            descriptor_names=["MolWt"],
            merging_approach=DescriptorMergingMethod.Concatenate,
        )
        assert "smiles_A_MolWt" in result.columns
        assert "smiles_B_MolWt" in result.columns


# ---------------------------------------------------------------------------
# Numeric sanity checks
# ---------------------------------------------------------------------------


class TestNumericSanity:
    def test_molwt_methane_single_monomer(self, single_monomer_df):
        """MolWt of methane (C) should be ≈ 16.04 for the first row."""
        result = compute_rdkit_descriptors(
            data=single_monomer_df,
            smiles_cols=["smiles_A"],
            descriptor_names=["MolWt"],
            merging_approach=DescriptorMergingMethod.NoMerging,
        )
        assert result.loc[0, "MolWt"] == MOLWT_METHANE

    def test_unequal_weights_differ_from_equal_weights(self, two_monomer_df, weights_col):
        """Unequal weights (60/40) must produce a different result than equal weights (50/50)."""
        result_unequal = compute_rdkit_descriptors(
            data=two_monomer_df,
            smiles_cols=["smiles_A", "smiles_B"],
            descriptor_names=["MolWt"],
            merging_approach=DescriptorMergingMethod.WeightedAverage,
            weights_col=weights_col,
        )

        df_equal = two_monomer_df.copy()
        df_equal["weight_A"] = 50.0
        df_equal["weight_B"] = 50.0
        result_equal = compute_rdkit_descriptors(
            data=df_equal,
            smiles_cols=["smiles_A", "smiles_B"],
            descriptor_names=["MolWt"],
            merging_approach=DescriptorMergingMethod.WeightedAverage,
            weights_col=weights_col,
        )

        # Row 0 has 60/40 split in the original fixture → values differ
        assert result_unequal.loc[0, "MolWt"] != pytest.approx(result_equal.loc[0, "MolWt"])

    def test_concatenate_preserves_individual_monomer_values(self, two_monomer_df):
        """Concatenated output must exactly match each monomer's individual NoMerging result."""
        # Individual monomer A descriptors
        df_a_only = two_monomer_df[["smiles_A", "Tg"]].copy()
        result_a = compute_rdkit_descriptors(
            data=df_a_only,
            smiles_cols=["smiles_A"],
            descriptor_names=["MolWt"],
            merging_approach=DescriptorMergingMethod.NoMerging,
        )

        result_concat = compute_rdkit_descriptors(
            data=two_monomer_df,
            smiles_cols=["smiles_A", "smiles_B"],
            descriptor_names=["MolWt"],
            merging_approach=DescriptorMergingMethod.Concatenate,
        )

        pd.testing.assert_series_equal(
            result_a["MolWt"].reset_index(drop=True),
            result_concat["smiles_A_MolWt"].reset_index(drop=True),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Index preservation
# ---------------------------------------------------------------------------


class TestIndexPreservation:
    def test_output_index_matches_input(self, two_monomer_df):
        result = compute_rdkit_descriptors(
            data=two_monomer_df,
            smiles_cols=["smiles_A", "smiles_B"],
            descriptor_names=DESCRIPTORS,
            merging_approach=DescriptorMergingMethod.Concatenate,
        )
        assert list(result.index) == list(two_monomer_df.index)

    def test_output_index_preserved_for_non_default_index(self):
        df = pd.DataFrame(
            {"smiles_A": ["C", "CC"], "Tg": [300.0, 350.0]}, index=[10, 20]
        )
        result = compute_rdkit_descriptors(
            data=df, smiles_cols=["smiles_A"], descriptor_names=["MolWt"],
            merging_approach=DescriptorMergingMethod.NoMerging,
        )
        assert list(result.index) == [10, 20]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_unknown_descriptor_name_is_skipped_with_warning(self, single_monomer_df, caplog):
        """Unknown descriptor names are silently skipped (logged as warning)."""
        import logging

        with caplog.at_level(logging.WARNING, logger="polynet.featurizer.descriptors"):
            result = compute_rdkit_descriptors(
                data=single_monomer_df,
                smiles_cols=["smiles_A"],
                descriptor_names=["MolWt", "NotARealDescriptor"],
                merging_approach=DescriptorMergingMethod.NoMerging,
            )
        assert "MolWt" in result.columns
        assert "NotARealDescriptor" not in result.columns
        assert any("NotARealDescriptor" in r.message for r in caplog.records)

    def test_weighted_average_without_weights_col_raises(self, two_monomer_df):
        """``build_vector_representation`` raises if WeightedAverage is requested without weights."""
        from polynet.config.enums import MolecularDescriptor
        from polynet.featurizer.descriptors import build_vector_representation

        with pytest.raises(ValueError, match="weights_col"):
            build_vector_representation(
                data=two_monomer_df,
                molecular_descriptors={MolecularDescriptor.RDKit: ["MolWt"]},
                smiles_cols=["smiles_A", "smiles_B"],
                id_col=None,
                target_col="Tg",
                merging_approach=DescriptorMergingMethod.WeightedAverage,
                weights_col=None,
            )
