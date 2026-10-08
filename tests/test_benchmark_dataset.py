"""
tests/test_benchmark_dataset.py
===============================
``data.benchmark_dataset`` lets the CLI use a built-in benchmark (e.g. the
curated Tg dataset) instead of a CSV file; exactly one of ``data_path`` and
``benchmark_dataset`` must be given. The benchmark is validated like a CSV.
"""

from unittest import mock

import pandas as pd
from pydantic import ValidationError
import pytest

from polynet.config.schemas import DataConfig
from polynet.data import loader

_BASE = dict(
    data_name="tg.csv",
    smiles_cols=["PSMILES"],
    target_variable_col="Tg(K)",
    problem_type="regression",
    num_classes=1,
)


def test_exactly_one_data_source_is_required():
    assert DataConfig(**_BASE, benchmark_dataset="curated_tg").data_path is None
    assert DataConfig(**_BASE, data_path="data.csv").benchmark_dataset is None
    for sources in ({}, {"data_path": "data.csv", "benchmark_dataset": "curated_tg"}):
        with pytest.raises(ValidationError, match="exactly one data source"):
            DataConfig(**_BASE, **sources)


def _fake_benchmark(df):
    creator = mock.MagicMock()
    creator.return_value.create_dataset.return_value = df
    return mock.patch("polynet.data.creator.DatasetCreator", creator)


def test_benchmark_dataset_is_loaded_and_indexed_by_id():
    df = pd.DataFrame({"ID": [0, 1], "PSMILES": ["[*]CC[*]", "[*]CO[*]"], "Tg(K)": [300.0, 350.0]})
    with _fake_benchmark(df):
        out = loader.load_benchmark_dataset("curated_tg", ["PSMILES"], "Tg(K)", "ID", "regression")
    assert out.index.name == "ID" and list(out["Tg(K)"]) == [300.0, 350.0]


def test_benchmark_dataset_is_validated_like_a_csv():
    df = pd.DataFrame({"ID": [0, 1], "PSMILES": ["[*]CC[*]", "[*]CO[*]"], "Tg(K)": [300.0, 350.0]})
    with _fake_benchmark(df), pytest.raises(ValueError):
        loader.load_benchmark_dataset("curated_tg", ["SMILES"], "Tg(K)", "ID", "regression")


def test_bundled_fluorine_nmr_snr_dataset_loads_and_validates():
    smiles_cols = [f"smiles_{i}" for i in range(1, 7)]
    df = loader.load_benchmark_dataset("fluorine_nmr_snr", smiles_cols, "SNR", "ID", "regression")
    assert len(df) == 418 and df.index.is_unique
    # Copolymers: the six molar ratios of every polymer sum to 1.
    ratios = df[[f"ratio_{i}" for i in range(1, 7)]].sum(axis=1)
    assert ratios.round(6).eq(1).all()
    # Missing measurements are empty (NaN), not placeholder strings.
    assert df["MolWt"].dtype.kind == "f" and df["MolWt"].notna().sum() == 159
