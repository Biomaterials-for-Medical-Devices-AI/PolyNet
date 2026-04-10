"""
tests/test_persistence.py
==========================
Tests for the column-validation logic in ``polynet.models.persistence``.

Strategy
--------
``load_dataframes`` reads real CSV files from disk, which makes integration
testing expensive. Instead, we:

1. Test ``_resolve_features`` directly — it's a pure function with no I/O.
2. Test ``load_dataframes`` with ``pd.read_csv`` and ``sanitise_df`` mocked
   so no actual files are needed.

Functions under test:
- ``_resolve_features``  (pure, no mocking needed)
- ``load_dataframes``    (mocked I/O)
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from polynet.config.enums import MolecularDescriptor
from polynet.models.persistence import _resolve_features, load_dataframes


# ---------------------------------------------------------------------------
# _resolve_features — pure function tests
# ---------------------------------------------------------------------------


class TestResolveFeatures:
    """
    ``_resolve_features`` returns:
    - a list of column names to validate against the CSV, or
    - ``None`` to signal that this descriptor should be skipped entirely.
    """

    # --- Auto-feature descriptors (read cols from CSV) ---

    @pytest.mark.parametrize(
        "descriptor",
        [
            MolecularDescriptor.PolyBERT,
            MolecularDescriptor.Morgan,
            MolecularDescriptor.RDKitFP,
            MolecularDescriptor.PolyMetriX,
        ],
    )
    def test_auto_descriptor_returns_non_target_columns(self, descriptor):
        """Auto-feature descriptors read column names from the CSV directly."""
        sanitised_cols = ["feat_0", "feat_1", "feat_2", "Tg"]
        result = _resolve_features(descriptor, [], sanitised_cols, "Tg")
        assert result == ["feat_0", "feat_1", "feat_2"]

    @pytest.mark.parametrize(
        "descriptor",
        [
            MolecularDescriptor.PolyBERT,
            MolecularDescriptor.Morgan,
            MolecularDescriptor.RDKitFP,
            MolecularDescriptor.PolyMetriX,
        ],
    )
    def test_auto_descriptor_excludes_target_from_result(self, descriptor):
        """Target column must never appear in the resolved feature list."""
        sanitised_cols = ["feat_0", "Tg"]
        result = _resolve_features(descriptor, [], sanitised_cols, "Tg")
        assert "Tg" not in result

    # --- RDKit: configured columns validated ---

    def test_rdkit_returns_configured_columns(self):
        features = ["MolWt", "TPSA"]
        sanitised_cols = ["MolWt", "TPSA", "Tg"]
        result = _resolve_features(MolecularDescriptor.RDKit, features, sanitised_cols, "Tg")
        assert result == ["MolWt", "TPSA"]

    def test_rdkit_empty_features_returns_none(self):
        """Empty feature list → skip this descriptor (return None)."""
        result = _resolve_features(MolecularDescriptor.RDKit, [], ["MolWt", "Tg"], "Tg")
        assert result is None

    def test_rdkit_none_features_returns_none(self):
        """None feature list → skip this descriptor (return None)."""
        result = _resolve_features(MolecularDescriptor.RDKit, None, ["MolWt", "Tg"], "Tg")
        assert result is None

    # --- DataFrame: configured columns validated ---

    def test_dataframe_returns_configured_columns(self):
        features = ["col1", "col2"]
        sanitised_cols = ["col1", "col2", "Tg"]
        result = _resolve_features(MolecularDescriptor.DataFrame, features, sanitised_cols, "Tg")
        assert result == ["col1", "col2"]

    def test_dataframe_empty_features_returns_none(self):
        result = _resolve_features(MolecularDescriptor.DataFrame, [], ["col1", "Tg"], "Tg")
        assert result is None


# ---------------------------------------------------------------------------
# load_dataframes — mocked I/O tests
# ---------------------------------------------------------------------------


def _make_representation_options(descriptor: MolecularDescriptor, features) -> MagicMock:
    opts = MagicMock()
    opts.molecular_descriptors = {descriptor: features}
    opts.weights_col = None
    return opts


def _make_data_options(target: str = "Tg", smiles_cols: list | None = None) -> MagicMock:
    opts = MagicMock()
    opts.target_variable_col = target
    opts.smiles_cols = smiles_cols or ["smiles"]
    return opts


class TestLoadDataframes:
    """
    ``load_dataframes`` is tested by patching:
    - ``polynet.models.persistence.pd.read_csv``   → returns a controlled DataFrame
    - ``polynet.models.persistence.sanitise_df``   → identity (returns its ``df`` arg)
    - ``polynet.models.persistence.representation_file`` → returns a dummy Path
    """

    @pytest.fixture
    def dummy_path(self, tmp_path) -> Path:
        return tmp_path / "experiment"

    def _run(self, representation_options, data_options, experiment_path, mock_df):
        """Patch I/O and run ``load_dataframes``."""
        with (
            patch("polynet.models.persistence.pd.read_csv", return_value=mock_df),
            patch(
                "polynet.models.persistence.sanitise_df", side_effect=lambda df, **_: df
            ),
            patch(
                "polynet.models.persistence.representation_file",
                return_value=Path("/fake/path.csv"),
            ),
        ):
            return load_dataframes(representation_options, data_options, experiment_path)

    # --- Happy paths ---

    def test_rdkit_valid_columns_returns_dataframe(self, dummy_path):
        """Valid RDKit CSV with all expected columns → descriptor is in result dict."""
        mock_df = pd.DataFrame({"MolWt": [1.0], "TPSA": [2.0], "Tg": [300.0]})
        rep_opts = _make_representation_options(MolecularDescriptor.RDKit, ["MolWt", "TPSA"])
        data_opts = _make_data_options()

        result = self._run(rep_opts, data_opts, dummy_path, mock_df)

        assert MolecularDescriptor.RDKit in result
        pd.testing.assert_frame_equal(result[MolecularDescriptor.RDKit], mock_df)

    def test_auto_descriptor_returns_dataframe(self, dummy_path):
        """Auto-feature descriptors (PolyBERT, Morgan, etc.) skip column validation."""
        mock_df = pd.DataFrame(
            {"polybert_0": [0.1], "polybert_1": [0.2], "Tg": [300.0]}
        )
        rep_opts = _make_representation_options(MolecularDescriptor.PolyBERT, [])
        data_opts = _make_data_options()

        result = self._run(rep_opts, data_opts, dummy_path, mock_df)

        assert MolecularDescriptor.PolyBERT in result

    def test_empty_features_skips_descriptor(self, dummy_path):
        """Empty feature list → descriptor is not added to the result dict."""
        mock_df = pd.DataFrame({"MolWt": [1.0], "Tg": [300.0]})
        rep_opts = _make_representation_options(MolecularDescriptor.RDKit, [])
        data_opts = _make_data_options()

        result = self._run(rep_opts, data_opts, dummy_path, mock_df)

        assert MolecularDescriptor.RDKit not in result

    # --- Error cases ---

    def test_missing_feature_columns_raises_value_error(self, dummy_path):
        """CSV missing expected columns → ``ValueError`` with informative message."""
        # CSV has MolWt but not TPSA
        mock_df = pd.DataFrame({"MolWt": [1.0], "Tg": [300.0]})
        rep_opts = _make_representation_options(
            MolecularDescriptor.RDKit, ["MolWt", "TPSA"]  # TPSA missing
        )
        data_opts = _make_data_options()

        with pytest.raises(ValueError, match="Missing expected feature columns"):
            self._run(rep_opts, data_opts, dummy_path, mock_df)

    def test_target_not_last_raises_value_error(self, dummy_path):
        """Target column not in last position → ``ValueError``."""
        # Target is in the middle, not last
        mock_df = pd.DataFrame({"Tg": [300.0], "MolWt": [1.0]})
        rep_opts = _make_representation_options(MolecularDescriptor.RDKit, ["MolWt"])
        data_opts = _make_data_options()

        with pytest.raises(ValueError, match="must be last"):
            self._run(rep_opts, data_opts, dummy_path, mock_df)

    def test_missing_and_wrong_order_shows_missing_error_first(self, dummy_path):
        """When both columns are missing and order is wrong, missing-columns error fires first."""
        mock_df = pd.DataFrame({"Tg": [300.0]})  # TPSA completely absent
        rep_opts = _make_representation_options(MolecularDescriptor.RDKit, ["TPSA"])
        data_opts = _make_data_options()

        with pytest.raises(ValueError, match="Missing expected feature columns"):
            self._run(rep_opts, data_opts, dummy_path, mock_df)
