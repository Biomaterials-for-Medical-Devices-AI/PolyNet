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


# --- subsets of large benchmarks (GUI) -------------------------------------------


def _numbered(n):
    return pd.DataFrame({"ID": range(n), "y": range(n)}, index=range(10, 10 + n))


def test_subsample_is_reproducible_nested_and_keeps_row_order():
    from polynet.data.creator import subsample_dataset

    df = _numbered(500)
    small, large = subsample_dataset(df, 50), subsample_dataset(df, 200)
    assert len(small) == 50 and len(large) == 200
    assert small.equals(subsample_dataset(df, 50))
    assert set(small.index) <= set(large.index)
    assert small.index.is_monotonic_increasing
    assert subsample_dataset(df, 500) is df and subsample_dataset(df, 900) is df
    with pytest.raises(ValueError):
        subsample_dataset(df, 0)


def _sample_size_app(n_rows: int):
    import pandas as pd
    import streamlit as st

    from polynet.app.components.forms.create_experiment import benchmark_sample_size_widget

    df = pd.DataFrame({"ID": range(n_rows)})
    st.session_state["n_used"] = len(benchmark_sample_size_widget(df))


def test_large_benchmark_defaults_to_a_subset_and_can_use_all():
    from streamlit.testing.v1 import AppTest

    from polynet.app.components.forms.create_experiment import DEFAULT_BENCHMARK_SAMPLES

    app = AppTest.from_function(_sample_size_app, kwargs={"n_rows": 7367}).run()
    assert app.session_state["n_used"] == DEFAULT_BENCHMARK_SAMPLES
    assert "7,367 polymers" in app.info[0].value

    app.number_input[0].set_value(7367).run()
    assert app.session_state["n_used"] == 7367


def test_small_benchmark_is_used_whole_without_a_widget():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_sample_size_app, kwargs={"n_rows": 418}).run()
    assert app.session_state["n_used"] == 418
    assert not app.number_input and not app.info


def _row_numbers_app():
    import pandas as pd
    import streamlit as st

    from polynet.app.components.forms.create_experiment import benchmark_sample_size_widget

    # "ID" holds each polymer's row in the full dataset; the index is not 0 … n − 1,
    # as a loader might return.
    full = pd.DataFrame({"ID": range(5000)}, index=range(100, 5100))
    subset = benchmark_sample_size_widget(full)
    st.session_state["rows"] = dict(zip(subset.index, subset["ID"]))


def test_subset_rows_keep_their_number_in_the_full_dataset():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_row_numbers_app).run()
    small = app.session_state["rows"]
    assert all(row == polymer for row, polymer in small.items())
    assert max(small) >= 1000  # not renumbered 0 … 999

    app.number_input[0].set_value(3000).run()
    large = app.session_state["rows"]
    assert all(row == polymer for row, polymer in large.items())
    assert set(small) <= set(large)


def test_held_out_rows_are_reproducible_and_never_in_the_subset():
    from polynet.data.creator import held_out_dataset, subsample_dataset

    df = _numbered(500)
    for n in (50, 200, 450):
        held_out = held_out_dataset(df, n, 100)
        assert not set(held_out.index) & set(subsample_dataset(df, n).index)
        assert held_out.equals(held_out_dataset(df, n, 100))
    assert len(held_out_dataset(df, 450, 100)) == 50  # only 50 left
    assert held_out_dataset(df, 500, 100).empty


def _download_app():
    import pandas as pd
    import streamlit as st

    from polynet.app.components.forms.create_experiment import benchmark_sample_size_widget

    subset = benchmark_sample_size_widget(pd.DataFrame({"ID": range(5000)}), "curated_tg")
    st.session_state["subset_ids"] = set(subset["ID"])


def test_large_benchmark_offers_unused_polymers_for_prediction():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_download_app).run()
    (button,) = app.get("download_button")
    assert "100 polymers not used" in button.proto.label
    assert button.proto.url  # the CSV itself is served by the media manager

    app.number_input[0].set_value(5000).run()
    assert not app.get("download_button")  # nothing left to hold out
