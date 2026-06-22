"""
tests/test_merging.py
=====================
Unit tests for the three descriptor-merging strategies.

These tests call the merging functions **directly** with manually constructed
``df_dict`` fixtures so they run in milliseconds without invoking RDKit,
PolyBERT, or any external library.

Functions under test (from ``polynet.featurizer.descriptors``):
- ``merge_weighted``
- ``_merge_concatenate``
- ``_single_smiles``
- ``_merge`` (dispatch function)
"""

import pandas as pd
import pytest

from polynet.config.enums import DescriptorMergingMethod
from polynet.featurizer.descriptors import (
    _merge,
    _merge_concatenate,
    _single_smiles,
    merge_weighted,
)

# ---------------------------------------------------------------------------
# merge_weighted
# ---------------------------------------------------------------------------


class TestMergeWeighted:
    """
    ``merge_weighted`` normalises each polymer's ratios by their per-row sum::

        result = Σ (weight_i / Σ_j weight_j) * descriptor_i

    When the ratios already sum to 100 (as in these fixtures) this is identical
    to the previous ``weight_i / 100`` behaviour, so these expected values are
    unchanged. See ``TestMergeWeightedNormalization`` for the non-100 cases.
    """

    @pytest.fixture
    def data_with_pct_weights(self) -> pd.DataFrame:
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
    def weights_col_map(self) -> dict[str, str]:
        return {"smiles_A": "weight_A", "smiles_B": "weight_B"}

    def test_column_values(
        self, known_df_dict, data_with_pct_weights, weights_col_map, empty_data_index
    ):
        """output = Σ (weight_i / 100) * descriptor_i per row."""
        result = merge_weighted(
            known_df_dict, data_with_pct_weights, weights_col_map, empty_data_index
        )

        # Row 0: 16*0.6 + 18*0.4 = 9.6 + 7.2 = 16.8
        assert result.loc[0, "MolWt"] == pytest.approx(16.8)
        # Row 1: 30*0.4 + 32*0.6 = 12.0 + 19.2 = 31.2
        assert result.loc[1, "MolWt"] == pytest.approx(31.2)
        # Row 2: equal weights → (44+46)/2 = 45.0
        assert result.loc[2, "MolWt"] == pytest.approx(45.0)

    def test_unequal_weights_differ_from_equal_weights(
        self, known_df_dict, data_with_pct_weights, weights_col_map, empty_data_index
    ):
        """Unequal weights (60/40) must produce a different result than equal weights (50/50)."""
        result_unequal = merge_weighted(
            known_df_dict, data_with_pct_weights, weights_col_map, empty_data_index
        )

        data_equal = data_with_pct_weights.copy()
        data_equal["weight_A"] = 50.0
        data_equal["weight_B"] = 50.0
        result_equal = merge_weighted(known_df_dict, data_equal, weights_col_map, empty_data_index)

        # Rows 0 and 1 have unequal weights in the original fixture
        assert result_unequal.loc[0, "MolWt"] != pytest.approx(result_equal.loc[0, "MolWt"])
        assert result_unequal.loc[1, "MolWt"] != pytest.approx(result_equal.loc[1, "MolWt"])

    def test_equal_weights_symmetric(self, known_df_dict, weights_col_map, empty_data_index):
        """50/50 weights on symmetric data: result should equal the per-monomer values scaled."""
        data_equal = pd.DataFrame(
            {
                "smiles_A": ["C", "CC", "CCC"],
                "smiles_B": ["O", "CO", "CCO"],
                "weight_A": [50.0, 50.0, 50.0],
                "weight_B": [50.0, 50.0, 50.0],
                "Tg": [300.0, 350.0, 375.0],
            }
        )
        result = merge_weighted(known_df_dict, data_equal, weights_col_map, empty_data_index)

        # Row 0: (16 * 0.5) + (18 * 0.5) = 8.0 + 9.0 = 17.0
        assert result.loc[0, "MolWt"] == pytest.approx(17.0)
        # Row 2: (44 * 0.5) + (46 * 0.5) = 22.0 + 23.0 = 45.0
        assert result.loc[2, "MolWt"] == pytest.approx(45.0)

    def test_preserves_index(
        self, known_df_dict, data_with_pct_weights, weights_col_map, empty_data_index
    ):
        result = merge_weighted(
            known_df_dict, data_with_pct_weights, weights_col_map, empty_data_index
        )
        assert list(result.index) == list(empty_data_index.index)

    def test_output_columns(
        self, known_df_dict, data_with_pct_weights, weights_col_map, empty_data_index
    ):
        result = merge_weighted(
            known_df_dict, data_with_pct_weights, weights_col_map, empty_data_index
        )
        assert set(result.columns) == {"MolWt", "HeavyAtomCount"}

    def test_output_shape(
        self, known_df_dict, data_with_pct_weights, weights_col_map, empty_data_index
    ):
        result = merge_weighted(
            known_df_dict, data_with_pct_weights, weights_col_map, empty_data_index
        )
        assert result.shape == (3, 2)


# ---------------------------------------------------------------------------
# _merge_concatenate
# ---------------------------------------------------------------------------


class TestMergeConcatenate:
    def test_column_count(self, known_df_dict, empty_data_index):
        """Output has 2× descriptors (one set per SMILES column)."""
        result = _merge_concatenate(known_df_dict, empty_data_index)
        assert len(result.columns) == 4  # 2 cols × 2 keys

    def test_column_names(self, known_df_dict, empty_data_index):
        """Columns are prefixed with the SMILES column name: ``<key>_<col>``."""
        result = _merge_concatenate(known_df_dict, empty_data_index)
        expected = {
            "smiles_A_MolWt",
            "smiles_A_HeavyAtomCount",
            "smiles_B_MolWt",
            "smiles_B_HeavyAtomCount",
        }
        assert set(result.columns) == expected

    def test_values_preserved(self, known_df_dict, empty_data_index):
        """Original values are preserved under the renamed columns."""
        result = _merge_concatenate(known_df_dict, empty_data_index)
        assert result.loc[0, "smiles_A_MolWt"] == pytest.approx(16.0)
        assert result.loc[0, "smiles_B_MolWt"] == pytest.approx(18.0)
        assert result.loc[2, "smiles_A_HeavyAtomCount"] == pytest.approx(3.0)

    def test_preserves_index(self, known_df_dict, empty_data_index):
        result = _merge_concatenate(known_df_dict, empty_data_index)
        assert list(result.index) == list(empty_data_index.index)

    def test_output_shape(self, known_df_dict, empty_data_index):
        result = _merge_concatenate(known_df_dict, empty_data_index)
        assert result.shape == (3, 4)

    def test_no_column_collision_after_prefixing(self, empty_data_index):
        """Same descriptor name across two keys does not collide after prefixing."""
        df_a = pd.DataFrame({"feat1": [1.0, 2.0]}, index=[0, 1])
        df_b = pd.DataFrame({"feat1": [3.0, 4.0]}, index=[0, 1])
        df_dict = {"key_A": df_a, "key_B": df_b}
        empty = pd.DataFrame(index=[0, 1])
        result = _merge_concatenate(df_dict, empty)
        assert "key_A_feat1" in result.columns
        assert "key_B_feat1" in result.columns
        assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# _single_smiles
# ---------------------------------------------------------------------------


class TestSingleSmiles:
    def test_single_key_passthrough(self, single_key_df_dict, empty_data_index):
        """For a single-key dict, output equals the input DataFrame."""
        result = _single_smiles(single_key_df_dict, empty_data_index)
        expected = single_key_df_dict["smiles_A"]
        pd.testing.assert_frame_equal(
            result[["MolWt", "HeavyAtomCount"]].reset_index(drop=True),
            expected.reset_index(drop=True),
        )

    def test_single_key_output_shape(self, single_key_df_dict, empty_data_index):
        result = _single_smiles(single_key_df_dict, empty_data_index)
        assert result.shape == (3, 2)

    def test_two_keys_concatenates_both_dfs(self, known_df_dict, empty_data_index):
        """With multiple keys, all DataFrames are concatenated column-wise."""
        result = _single_smiles(known_df_dict, empty_data_index)
        # Both DFs have 2 cols → 4 total (column names will duplicate)
        assert result.shape[1] == 4

    def test_preserves_index(self, single_key_df_dict, empty_data_index):
        result = _single_smiles(single_key_df_dict, empty_data_index)
        assert list(result.index) == list(empty_data_index.index)


# ---------------------------------------------------------------------------
# _merge (dispatch)
# ---------------------------------------------------------------------------


class TestMergeDispatch:
    """Verify that ``_merge`` routes to the correct strategy."""

    def test_dispatch_concatenate(self, known_df_dict, two_monomer_df, empty_data_index):
        direct = _merge_concatenate(known_df_dict, empty_data_index)
        via_dispatch = _merge(
            known_df_dict,
            two_monomer_df,
            None,
            empty_data_index,
            DescriptorMergingMethod.Concatenate,
        )
        pd.testing.assert_frame_equal(direct, via_dispatch)

    def test_dispatch_no_merging(self, single_key_df_dict, single_monomer_df, empty_data_index):
        direct = _single_smiles(single_key_df_dict, empty_data_index)
        via_dispatch = _merge(
            single_key_df_dict,
            single_monomer_df,
            None,
            empty_data_index,
            DescriptorMergingMethod.NoMerging,
        )
        pd.testing.assert_frame_equal(direct, via_dispatch)

    def test_dispatch_weighted_average(self, known_df_dict, two_monomer_df, empty_data_index):
        weights_col = {"smiles_A": "weight_A", "smiles_B": "weight_B"}
        direct = merge_weighted(known_df_dict, two_monomer_df, weights_col, empty_data_index)
        via_dispatch = _merge(
            known_df_dict,
            two_monomer_df,
            weights_col,
            empty_data_index,
            DescriptorMergingMethod.WeightedAverage,
        )
        pd.testing.assert_frame_equal(direct, via_dispatch)

    def test_unsupported_method_raises(self, known_df_dict, two_monomer_df, empty_data_index):
        """An unknown merging method value raises ``ValueError``."""
        with pytest.raises(ValueError, match="Unsupported merging method"):
            _merge(known_df_dict, two_monomer_df, None, empty_data_index, "not_a_method")

    def test_accepts_enum_value(self, known_df_dict, two_monomer_df, empty_data_index):
        """``_merge`` accepts a ``DescriptorMergingMethod`` enum value without error."""
        result = _merge(
            known_df_dict,
            two_monomer_df,
            None,
            empty_data_index,
            DescriptorMergingMethod.Concatenate,
        )
        assert result is not None
        assert not result.empty


# ---------------------------------------------------------------------------
# merge_weighted — per-polymer ratio normalisation (sum-to-1)
# ---------------------------------------------------------------------------


class TestMergeWeightedNormalization:
    """
    ``merge_weighted`` divides each monomer's ratio by the per-row sum of the
    participating ratios, so the weights sum to 1 per polymer regardless of the
    input scale. This reproduces the old ``ratio / 100`` results exactly when
    the ratios already sum to 100, and handles any other scale.
    """

    @pytest.fixture
    def weights_col_map(self) -> dict[str, str]:
        return {"smiles_A": "weight_A", "smiles_B": "weight_B"}

    def test_fraction_scale_matches_percentage_scale(
        self, known_df_dict, weights_col_map, empty_data_index
    ):
        """0.6/0.4 (fractions) gives the same result as 60/40 (percentages)."""
        data_frac = pd.DataFrame(
            {"weight_A": [0.6, 0.4, 0.5], "weight_B": [0.4, 0.6, 0.5]}, index=[0, 1, 2]
        )
        data_pct = pd.DataFrame(
            {"weight_A": [60.0, 40.0, 50.0], "weight_B": [40.0, 60.0, 50.0]}, index=[0, 1, 2]
        )
        res_frac = merge_weighted(known_df_dict, data_frac, weights_col_map, empty_data_index)
        res_pct = merge_weighted(known_df_dict, data_pct, weights_col_map, empty_data_index)
        pd.testing.assert_frame_equal(res_frac, res_pct)

    def test_non_normalised_ratios(self, known_df_dict, weights_col_map, empty_data_index):
        """Raw ratios that do not sum to 100 are normalised by their row sum."""
        data = pd.DataFrame(
            {"weight_A": [1.0, 3.0, 1.0], "weight_B": [1.0, 1.0, 3.0]}, index=[0, 1, 2]
        )
        result = merge_weighted(known_df_dict, data, weights_col_map, empty_data_index)
        # Row 0: 1:1 → (16+18)/2 = 17.0
        assert result.loc[0, "MolWt"] == pytest.approx(17.0)
        # Row 1: 3:1 → (30*0.75 + 32*0.25) = 22.5 + 8.0 = 30.5
        assert result.loc[1, "MolWt"] == pytest.approx(30.5)
        # Row 2: 1:3 → (44*0.25 + 46*0.75) = 11.0 + 34.5 = 45.5
        assert result.loc[2, "MolWt"] == pytest.approx(45.5)

    def test_homopolymer_invariant_to_zero_ratio_column(
        self, known_df_dict, weights_col_map, empty_data_index
    ):
        """A 100/0 row equals monomer A alone (the zero-ratio block drops out)."""
        data = pd.DataFrame({"weight_A": [100.0], "weight_B": [0.0]}, index=[0])
        df_dict = {
            "smiles_A": known_df_dict["smiles_A"].loc[[0]],
            "smiles_B": known_df_dict["smiles_B"].loc[[0]],
        }
        result = merge_weighted(df_dict, data, weights_col_map, pd.DataFrame(index=[0]))
        assert result.loc[0, "MolWt"] == pytest.approx(16.0)

    def test_three_monomers(self, empty_data_index):
        """Three-monomer merge with ratios 1:1:2 normalises to 0.25/0.25/0.5."""
        idx = [0]
        df_dict = {
            "smiles_A": pd.DataFrame({"MolWt": [10.0]}, index=idx),
            "smiles_B": pd.DataFrame({"MolWt": [20.0]}, index=idx),
            "smiles_C": pd.DataFrame({"MolWt": [40.0]}, index=idx),
        }
        data = pd.DataFrame({"weight_A": [1.0], "weight_B": [1.0], "weight_C": [2.0]}, index=idx)
        weights_col = {"smiles_A": "weight_A", "smiles_B": "weight_B", "smiles_C": "weight_C"}
        result = merge_weighted(df_dict, data, weights_col, pd.DataFrame(index=idx))
        # 10*0.25 + 20*0.25 + 40*0.5 = 2.5 + 5.0 + 20.0 = 27.5
        assert result.loc[0, "MolWt"] == pytest.approx(27.5)
