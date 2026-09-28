"""
tests/test_fingerprint_settings.py
==================================
Morgan / RDKit count-fingerprint settings (``fp_size``, Morgan ``radius``)
are configurable; the defaults reproduce PolyNet's historical fingerprints,
which used RDKit's generator defaults (Morgan radius 3, 2048 bins).
"""

import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator

from polynet.config.enums import DescriptorMergingMethod, MolecularDescriptor
from polynet.config.schemas import RepresentationConfig
from polynet.config.schemas.fingerprints import (
    MorganFingerprintConfig,
    RDKitFingerprintConfig,
    resolve_fingerprint_config,
)
from polynet.featurizer.descriptors import (
    build_vector_representation,
    get_morgan_fingerprints,
    get_rdkitfp_fingerprints,
)

SMILES = ["CC(=O)OCCCOCC(C)(C)COCCCOC(=O)C=C", "C=CC(=O)OCC(F)(F)C(F)(F)F", "c1ccccc1O"]

# ---------------------------------------------------------------------------
# Settings schemas
# ---------------------------------------------------------------------------


def test_schema_defaults_are_rdkit_generator_defaults():
    assert MorganFingerprintConfig().model_dump() == {"fp_size": 2048, "radius": 3}
    assert RDKitFingerprintConfig().model_dump() == {"fp_size": 2048}


@pytest.mark.parametrize("value", [True, None, [], {}])
def test_legacy_values_select_the_defaults(value):
    assert resolve_fingerprint_config("morgan", value) == MorganFingerprintConfig()
    assert resolve_fingerprint_config("rdkitfp", value) == RDKitFingerprintConfig()


def test_partial_override_keeps_other_defaults():
    assert resolve_fingerprint_config("morgan", {"radius": 2}).model_dump() == {
        "fp_size": 2048,
        "radius": 2,
    }


@pytest.mark.parametrize(
    "descriptor, value, match",
    [
        ("morgan", {"bits": 1024}, "bits"),
        ("rdkitfp", {"radius": 2}, "radius"),
        ("morgan", {"fp_size": 0}, "fp_size"),
        ("morgan", {"radius": -1}, "radius"),
        ("morgan", {"fp_size": 10.5}, "fp_size"),
        ("morgan", "yes", "must be true or a mapping"),
    ],
)
def test_invalid_settings_are_rejected(descriptor, value, match):
    with pytest.raises(ValueError, match=match):
        resolve_fingerprint_config(descriptor, value)


def test_representation_config_validates_and_resolves_settings():
    cfg = RepresentationConfig(
        smiles_merge_approach="concatenate",
        molecular_descriptors={"morgan": {"fp_size": 1024, "radius": 2}, "rdkitfp": True},
    )
    # Stored fully resolved, so representation_options.json records the settings used.
    assert cfg.molecular_descriptors[MolecularDescriptor.Morgan] == {"fp_size": 1024, "radius": 2}
    assert cfg.molecular_descriptors[MolecularDescriptor.RDKitFP] == {"fp_size": 2048}
    # Re-validating the saved form gives the same config.
    assert RepresentationConfig.model_validate(cfg.model_dump()) == cfg
    with pytest.raises(ValidationError, match="radius"):
        RepresentationConfig(
            smiles_merge_approach="concatenate", molecular_descriptors={"rdkitfp": {"radius": 2}}
        )


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def _count(gen, smiles):
    return gen.GetCountFingerprintAsNumPy(MolFromSmiles(smiles)).astype(int).tolist()


def test_defaults_match_rdkit_generator_defaults():
    """PolyNet used to call the generators with no arguments; the defaults must
    reproduce those fingerprints exactly."""
    morgan, rdkitfp = get_morgan_fingerprints(SMILES), get_rdkitfp_fingerprints(SMILES)
    for s in SMILES:
        assert morgan[s] == _count(rdFingerprintGenerator.GetMorganGenerator(), s)
        assert rdkitfp[s] == _count(rdFingerprintGenerator.GetRDKitFPGenerator(), s)


def test_custom_settings_change_length_and_radius():
    fps = get_morgan_fingerprints(SMILES, fp_size=512, radius=2)
    for s in SMILES:
        assert len(fps[s]) == 512
        assert fps[s] == _count(rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512), s)
    assert len(get_rdkitfp_fingerprints(SMILES, fp_size=256)[SMILES[0]]) == 256


def _vectors(molecular_descriptors):
    data = pd.DataFrame({"id": [0, 1, 2], "smiles": SMILES, "y": [1.0, 2.0, 3.0]})
    return build_vector_representation(
        data=data,
        molecular_descriptors=molecular_descriptors,
        smiles_cols=["smiles"],
        id_col="id",
        target_col="y",
        merging_approach=DescriptorMergingMethod.NoMerging,
    )


def test_pipeline_uses_the_configured_settings():
    out = _vectors({MolecularDescriptor.Morgan: {"fp_size": 128, "radius": 1},
                    MolecularDescriptor.RDKitFP: {"fp_size": 64}})
    morgan_cols = [c for c in out[MolecularDescriptor.Morgan].columns if "morgan" in c]
    rdkit_cols = [c for c in out[MolecularDescriptor.RDKitFP].columns if "rdkitfp" in c]
    assert (len(morgan_cols), len(rdkit_cols)) == (128, 64)


def test_pipeline_defaults_are_unchanged_for_legacy_true():
    legacy = _vectors({MolecularDescriptor.Morgan: True})[MolecularDescriptor.Morgan]
    explicit = _vectors({MolecularDescriptor.Morgan: {"fp_size": 2048, "radius": 3}})[
        MolecularDescriptor.Morgan
    ]
    pd.testing.assert_frame_equal(legacy, explicit)
    cols = [c for c in legacy.columns if "morgan" in c]
    expected = np.array([_count(rdFingerprintGenerator.GetMorganGenerator(), s) for s in SMILES])
    np.testing.assert_array_equal(legacy[cols].to_numpy(), expected)
