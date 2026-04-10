"""
tests/test_config_loader.py
============================
Tests for the config normalisation pipeline in ``polynet.config._loader``.

Covers:
- Deprecated field migration (rdkit_descriptors, df_descriptors, polybert_fp)
- Enum value compatibility maps (merging methods, string representation)
- Field renames (node_feats → node_features, edge_feats → edge_features)
- Unrecognised top-level section handling in ``build_experiment_config``

No file I/O is needed — all tests call normalisation functions directly with
plain dicts.
"""

import warnings

import pytest

from polynet.config._loader import _normalise_representation, build_experiment_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _suppress_deprecation(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) and suppress all DeprecationWarnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Deprecated field migration
# ---------------------------------------------------------------------------


class TestDeprecatedFieldMigration:
    """rdkit_descriptors / df_descriptors / polybert_fp are stripped and
    folded into ``molecular_descriptors``."""

    def test_rdkit_descriptors_migrates_to_molecular_descriptors(self):
        raw = {"rdkit_descriptors": ["MolWt", "TPSA"]}
        result = _suppress_deprecation(_normalise_representation, raw)

        assert "rdkit_descriptors" not in result
        assert result["molecular_descriptors"]["rdkit"] == ["MolWt", "TPSA"]

    def test_rdkit_descriptors_emits_deprecation_warning(self):
        raw = {"rdkit_descriptors": ["MolWt"]}
        with pytest.warns(DeprecationWarning, match="rdkit_descriptors"):
            _normalise_representation(raw)

    def test_rdkit_descriptors_does_not_overwrite_existing_rdkit_key(self):
        """If ``molecular_descriptors.rdkit`` is already set, deprecated field is ignored."""
        raw = {
            "rdkit_descriptors": ["OldDescriptor"],
            "molecular_descriptors": {"rdkit": ["NewDescriptor"]},
        }
        result = _suppress_deprecation(_normalise_representation, raw)
        assert result["molecular_descriptors"]["rdkit"] == ["NewDescriptor"]

    def test_df_descriptors_migrates_to_molecular_descriptors(self):
        raw = {"df_descriptors": ["col1", "col2"]}
        result = _suppress_deprecation(_normalise_representation, raw)

        assert "df_descriptors" not in result
        assert result["molecular_descriptors"]["dataframe"] == ["col1", "col2"]

    def test_df_descriptors_emits_deprecation_warning(self):
        raw = {"df_descriptors": ["col1"]}
        with pytest.warns(DeprecationWarning, match="df_descriptors"):
            _normalise_representation(raw)

    def test_polybert_fp_migrates_to_molecular_descriptors(self):
        raw = {"polybert_fp": True}
        result = _suppress_deprecation(_normalise_representation, raw)

        assert "polybert_fp" not in result
        assert result["molecular_descriptors"]["polybert"] is True

    def test_polybert_fp_emits_deprecation_warning(self):
        raw = {"polybert_fp": True}
        with pytest.warns(DeprecationWarning, match="polybert_fp"):
            _normalise_representation(raw)

    def test_empty_deprecated_field_is_stripped_without_side_effect(self):
        """An empty/falsy deprecated field must not stomp on existing data."""
        raw = {
            "rdkit_descriptors": [],  # falsy
            "molecular_descriptors": {"rdkit": ["MolWt"]},
        }
        result = _suppress_deprecation(_normalise_representation, raw)
        assert result["molecular_descriptors"]["rdkit"] == ["MolWt"]

    def test_multiple_deprecated_fields_all_migrated(self):
        raw = {
            "rdkit_descriptors": ["MolWt"],
            "df_descriptors": ["col1"],
        }
        result = _suppress_deprecation(_normalise_representation, raw)
        assert result["molecular_descriptors"]["rdkit"] == ["MolWt"]
        assert result["molecular_descriptors"]["dataframe"] == ["col1"]


# ---------------------------------------------------------------------------
# Merging method compatibility
# ---------------------------------------------------------------------------


class TestMergingCompat:
    """Legacy human-readable merging values must be remapped to enum values."""

    @pytest.mark.parametrize(
        "legacy, expected",
        [
            ("Weighted Average", "weighted_average"),
            ("Concatenate", "concatenate"),
            ("No Merging", "no_merging"),
        ],
    )
    def test_single_string_remapped(self, legacy, expected):
        raw = {"smiles_merge_approach": legacy}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = _normalise_representation(raw)
        assert result["smiles_merge_approach"] == [expected]

    def test_already_correct_value_passthrough(self):
        raw = {"smiles_merge_approach": "weighted_average"}
        result = _normalise_representation(raw)
        assert result["smiles_merge_approach"] == ["weighted_average"]

    def test_string_is_wrapped_in_list(self):
        """smiles_merge_approach is always normalised to a list."""
        raw = {"smiles_merge_approach": "concatenate"}
        result = _normalise_representation(raw)
        assert isinstance(result["smiles_merge_approach"], list)

    def test_list_values_are_individually_remapped(self):
        """Each item in a list value is remapped independently."""
        raw = {"smiles_merge_approach": ["Weighted Average", "Concatenate"]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = _normalise_representation(raw)
        assert result["smiles_merge_approach"] == ["weighted_average", "concatenate"]


# ---------------------------------------------------------------------------
# Field renames
# ---------------------------------------------------------------------------


class TestFieldRenames:
    def test_node_feats_renamed_to_node_features(self):
        raw = {"node_feats": ["GetAtomicNum"]}
        result = _normalise_representation(raw)
        assert "node_feats" not in result
        assert result["node_features"] == ["GetAtomicNum"]

    def test_edge_feats_renamed_to_edge_features(self):
        raw = {"edge_feats": ["GetBondTypeAsDouble"]}
        result = _normalise_representation(raw)
        assert "edge_feats" not in result
        assert result["edge_features"] == ["GetBondTypeAsDouble"]

    def test_already_correct_keys_passthrough(self):
        raw = {"node_features": ["GetAtomicNum"], "edge_features": []}
        result = _normalise_representation(raw)
        assert result["node_features"] == ["GetAtomicNum"]
        assert result["edge_features"] == []


# ---------------------------------------------------------------------------
# build_experiment_config — top-level section handling
# ---------------------------------------------------------------------------


class TestBuildExperimentConfig:
    def test_unrecognised_top_level_key_raises_value_error(self):
        """Completely unknown top-level section keys raise ``ValueError``."""
        with pytest.raises(ValueError, match="Unrecognised"):
            build_experiment_config({"totally_unknown_section": {}})

    def test_legacy_section_key_accepted(self):
        """Legacy keys like ``RepresentationOptions`` are accepted by the normaliser."""
        # We only need to verify it doesn't raise for an unrecognised key;
        # the actual schema validation is Pydantic's job and is tested elsewhere.
        # Call with only the legacy key to confirm it's in _SECTION_NORMALISERS.
        from polynet.config._loader import _SECTION_NORMALISERS

        assert "RepresentationOptions" in _SECTION_NORMALISERS
        assert "DataOptions" in _SECTION_NORMALISERS
        assert "GeneralConfigOptions" in _SECTION_NORMALISERS

    def test_duplicate_legacy_and_canonical_key_raises(self):
        """Using both a legacy key and its canonical equivalent raises ``ValueError``."""
        with pytest.raises(ValueError, match="Duplicate section"):
            build_experiment_config(
                {
                    "representation": {"molecular_descriptors": {}},
                    "RepresentationOptions": {"molecular_descriptors": {}},
                }
            )
