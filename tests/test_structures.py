"""
tests/test_structures.py
========================
One Streamlit-free structure preparation (detect → validate → canonicalise)
shared by the GUI, the CLI and ``predict_external``; duplicate-ID checks.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from polynet.config.enums import StringRepresentation
from polynet.data.loader import load_dataset
from polynet.data.structures import (
    canonicalise_structures,
    detect_string_representation,
    find_invalid_structures,
    prepare_structures,
)
from polynet.utils.validation import find_duplicate_ids


def test_detects_smiles_and_psmiles():
    assert detect_string_representation(pd.DataFrame({"s": ["CCO", "c1ccccc1"]}), ["s"]) == "smiles"
    assert detect_string_representation(pd.DataFrame({"s": ["[*]CC[*]", "[*]OC[*]"]}), ["s"]) == "psmiles"


def test_canonicalises_smiles():
    df = pd.DataFrame({"s": ["OCC", "C(C)O", "c1ccccc1"]})
    out, rep = prepare_structures(df, ["s"])
    assert rep == StringRepresentation.SMILES
    assert out["s"].tolist() == ["CCO", "CCO", "c1ccccc1"]
    assert df["s"].tolist() == ["OCC", "C(C)O", "c1ccccc1"]  # input not modified


def test_canonicalise_can_be_switched_off():
    df = pd.DataFrame({"s": ["OCC"]})
    out, _ = prepare_structures(df, ["s"], canonicalise=False)
    assert out["s"].tolist() == ["OCC"]


def test_invalid_structures_raise_with_examples_per_column():
    df = pd.DataFrame({"a": ["CCO", "C1CC", "xyz"], "b": ["CCN", "CC", "(("]})
    with pytest.raises(ValueError) as err:
        prepare_structures(df, ["a", "b"])
    msg = str(err.value)
    assert "column 'a': 2 invalid, e.g. C1CC, xyz" in msg
    assert "column 'b': 1 invalid, e.g. ((" in msg


def test_missing_structures_rejected_by_default_allowed_on_request(caplog):
    df = pd.DataFrame({"a": ["OCC", "CCN"], "b": ["CC", np.nan]})
    with pytest.raises(ValueError, match="column 'b': 1 invalid"):
        prepare_structures(df, ["a", "b"])
    with caplog.at_level(logging.WARNING):
        out, _ = prepare_structures(df, ["a", "b"], allow_missing=True)
    assert out["a"].tolist() == ["CCO", "CCN"] and pd.isna(out["b"].iloc[1])
    assert "1 missing value" in caplog.text


def test_declared_representation_wins_with_a_warning(caplog):
    df = pd.DataFrame({"s": ["[*]CC[*]", "[*]OC[*]"]})
    with caplog.at_level(logging.WARNING):
        _, rep = prepare_structures(df, ["s"], representation="smiles", canonicalise=False)
    assert rep == StringRepresentation.SMILES
    assert "look like psmiles" in caplog.text


def test_find_invalid_is_empty_for_valid_data():
    df = pd.DataFrame({"s": ["CCO", "[*]CC[*]"]})
    assert find_invalid_structures(df, ["s"], StringRepresentation.SMILES) == {}


def test_canonicalisation_failure_is_reported():
    df = pd.DataFrame({"s": ["CCO", "not-a-smiles"]})
    with pytest.raises(ValueError, match=r"(?s)Could not canonicalise.*not-a-smiles"):
        canonicalise_structures(df, ["s"], StringRepresentation.SMILES)


# ---------------------------------------------------------------------------
# Duplicate IDs
# ---------------------------------------------------------------------------


def test_find_duplicate_ids():
    assert find_duplicate_ids([1, 2, 2, 3, 1, 2]) == [2, 1]
    assert find_duplicate_ids(["a", "b"]) == []


def test_cli_loader_rejects_duplicate_ids_with_examples(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame({"id": [1, 2, 2, 5, 5], "smiles": ["CCO"] * 5, "y": range(5)}).to_csv(path, index=False)
    with pytest.raises(ValueError, match=r"2 duplicated value\(s\), e.g. 2, 5"):
        load_dataset(path, smiles_cols=["smiles"], target_col="y", id_col="id")
