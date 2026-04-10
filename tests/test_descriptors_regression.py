"""
tests/test_descriptors_regression.py
======================================
Regression tests for ``compute_rdkit_descriptors`` against user-supplied
reference outputs.

These tests lock in the exact numerical values produced by each merging
strategy so that future refactors are caught immediately if they change
descriptor values.

Fixture files (``tests/fixtures/``)
-------------------------------------
input_polymers.csv
    Input dataset.
    Columns: smiles_A, smiles_B, weight_A, weight_B, LogF_SA
    Weight columns are in **percentage** scale (0–100).

expected_weighted_average.csv
    Output of WeightedAverage merging with weights_col mapped to weight_A / weight_B.
    Columns: MolWt, TPSA, NumRotatableBonds

expected_concatenate.csv
    Output of Concatenate merging.
    Columns: smiles_A_MolWt, smiles_A_TPSA, smiles_A_NumRotatableBonds,
             smiles_B_MolWt, smiles_B_TPSA, smiles_B_NumRotatableBonds

expected_no_merging.csv
    Output of NoMerging using only smiles_A.
    Columns: MolWt, TPSA, NumRotatableBonds

All files are read with ``index_col=0`` and were written with ``df.to_csv()``.
"""

from pathlib import Path

import pandas as pd
import pytest

from polynet.config.enums import DescriptorMergingMethod
from polynet.featurizer import compute_rdkit_descriptors

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"
DESCRIPTORS = ["MolWt", "TPSA", "NumRotatableBonds"]
SMILES_COLS = ["smiles_A", "smiles_B"]
WEIGHTS_COL = {"smiles_A": "weight_A", "smiles_B": "weight_B"}
TARGET_COL = "LogF_SA"


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def input_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "input_polymers.csv", index_col=0)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _load_expected(filename: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES / filename, index_col=0)


# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------


class TestWeightedAverageRegression:
    def test_output_matches_reference(self, input_df):
        result = compute_rdkit_descriptors(
            data=input_df,
            smiles_cols=SMILES_COLS,
            descriptor_names=DESCRIPTORS,
            merging_approach=DescriptorMergingMethod.WeightedAverage,
            weights_col=WEIGHTS_COL,
        )
        expected = _load_expected("expected_weighted_average.csv")

        pd.testing.assert_frame_equal(result, expected, check_like=True, check_dtype=False)

    def test_column_names(self, input_df):
        result = compute_rdkit_descriptors(
            data=input_df,
            smiles_cols=SMILES_COLS,
            descriptor_names=DESCRIPTORS,
            merging_approach=DescriptorMergingMethod.WeightedAverage,
            weights_col=WEIGHTS_COL,
        )
        assert set(result.columns) == set(DESCRIPTORS)

    def test_row_count(self, input_df):
        result = compute_rdkit_descriptors(
            data=input_df,
            smiles_cols=SMILES_COLS,
            descriptor_names=DESCRIPTORS,
            merging_approach=DescriptorMergingMethod.WeightedAverage,
            weights_col=WEIGHTS_COL,
        )
        assert len(result) == len(input_df)

    def test_no_nulls(self, input_df):
        result = compute_rdkit_descriptors(
            data=input_df,
            smiles_cols=SMILES_COLS,
            descriptor_names=DESCRIPTORS,
            merging_approach=DescriptorMergingMethod.WeightedAverage,
            weights_col=WEIGHTS_COL,
        )
        assert not result.isnull().any().any()


class TestConcatenateRegression:
    @pytest.fixture(scope="class")
    def result(self, input_df):
        return compute_rdkit_descriptors(
            data=input_df,
            smiles_cols=SMILES_COLS,
            descriptor_names=DESCRIPTORS,
            merging_approach=DescriptorMergingMethod.Concatenate,
        )

    def test_output_matches_reference(self, result):
        expected = _load_expected("expected_concatenate.csv")
        pd.testing.assert_frame_equal(result, expected, check_like=True, check_dtype=False)

    def test_column_names(self, result):
        expected_cols = {
            f"{col}_{desc}" for col in SMILES_COLS for desc in DESCRIPTORS
        }
        assert set(result.columns) == expected_cols

    def test_row_count(self, result, input_df):
        assert len(result) == len(input_df)

    def test_no_nulls(self, result):
        assert not result.isnull().any().any()

    def test_monomer_a_values_independent_of_monomer_b(self, result, input_df):
        """smiles_A columns must equal the NoMerging result for smiles_A alone."""
        no_merge = compute_rdkit_descriptors(
            data=input_df,
            smiles_cols=["smiles_A"],
            descriptor_names=DESCRIPTORS,
            merging_approach=DescriptorMergingMethod.NoMerging,
        )
        for desc in DESCRIPTORS:
            pd.testing.assert_series_equal(
                result[f"smiles_A_{desc}"].reset_index(drop=True),
                no_merge[desc].reset_index(drop=True),
                check_names=False,
                rtol=1e-5,
            )


class TestNoMergingRegression:
    @pytest.fixture(scope="class")
    def result(self, input_df):
        return compute_rdkit_descriptors(
            data=input_df,
            smiles_cols=["smiles_A"],
            descriptor_names=DESCRIPTORS,
            merging_approach=DescriptorMergingMethod.NoMerging,
        )

    def test_output_matches_reference(self, result):
        expected = _load_expected("expected_no_merging.csv")
        pd.testing.assert_frame_equal(result, expected, check_like=True, check_dtype=False)

    def test_column_names(self, result):
        assert set(result.columns) == set(DESCRIPTORS)

    def test_row_count(self, result, input_df):
        assert len(result) == len(input_df)

    def test_no_nulls(self, result):
        assert not result.isnull().any().any()


# ---------------------------------------------------------------------------
# Cross-strategy consistency
# ---------------------------------------------------------------------------


class TestCrossStrategyConsistency:
    """Properties that must hold regardless of which strategy is used."""

    def test_concatenate_smiles_a_matches_no_merging(self, input_df):
        """The smiles_A block in Concatenate must equal the NoMerging result for smiles_A."""
        concat = compute_rdkit_descriptors(
            data=input_df,
            smiles_cols=SMILES_COLS,
            descriptor_names=DESCRIPTORS,
            merging_approach=DescriptorMergingMethod.Concatenate,
        )
        no_merge = compute_rdkit_descriptors(
            data=input_df,
            smiles_cols=["smiles_A"],
            descriptor_names=DESCRIPTORS,
            merging_approach=DescriptorMergingMethod.NoMerging,
        )
        for desc in DESCRIPTORS:
            pd.testing.assert_series_equal(
                concat[f"smiles_A_{desc}"].reset_index(drop=True),
                no_merge[desc].reset_index(drop=True),
                check_names=False,
                rtol=1e-5,
            )

    def test_all_strategies_return_same_row_count(self, input_df):
        n = len(input_df)
        for approach, kw in [
            (DescriptorMergingMethod.WeightedAverage, {"weights_col": WEIGHTS_COL}),
            (DescriptorMergingMethod.Concatenate, {}),
            (DescriptorMergingMethod.NoMerging, {}),
        ]:
            smiles = SMILES_COLS if approach != DescriptorMergingMethod.NoMerging else ["smiles_A"]
            result = compute_rdkit_descriptors(
                data=input_df,
                smiles_cols=smiles,
                descriptor_names=DESCRIPTORS,
                merging_approach=approach,
                **kw,
            )
            assert len(result) == n, f"Row count mismatch for {approach}"

    def test_weighted_average_pure_component_equals_no_merging(self, input_df):
        """
        When weight_A=100 and weight_B=0, WeightedAverage for smiles_A must
        equal NoMerging for smiles_A.
        """
        df_pure = input_df.copy()
        df_pure["weight_A"] = 100.0
        df_pure["weight_B"] = 0.0

        weighted = compute_rdkit_descriptors(
            data=df_pure,
            smiles_cols=SMILES_COLS,
            descriptor_names=DESCRIPTORS,
            merging_approach=DescriptorMergingMethod.WeightedAverage,
            weights_col=WEIGHTS_COL,
        )
        no_merge = compute_rdkit_descriptors(
            data=input_df,
            smiles_cols=["smiles_A"],
            descriptor_names=DESCRIPTORS,
            merging_approach=DescriptorMergingMethod.NoMerging,
        )
        for desc in DESCRIPTORS:
            pd.testing.assert_series_equal(
                weighted[desc].reset_index(drop=True),
                no_merge[desc].reset_index(drop=True),
                check_names=False,
                check_dtype=False,
                rtol=1e-5,
            )
