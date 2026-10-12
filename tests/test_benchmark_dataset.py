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


# --- benchmark subsets and external sets (GUI) -----------------------------------


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

    st.session_state.pop("n_used", None)  # set again only if the widget returns
    df = pd.DataFrame({"ID": range(n_rows)})
    st.session_state["n_used"] = len(benchmark_sample_size_widget(df, "bench"))


def test_large_benchmark_defaults_to_1000_with_a_note_on_training_time():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_sample_size_app, kwargs={"n_rows": 7367}).run()
    assert app.session_state["n_used"] == 1000
    assert "7,367 polymers" in app.info[0].value and "hours" in app.info[0].value
    assert "100 polymers not used" in app.get("download_button")[0].proto.label


def test_small_benchmark_defaults_to_all_but_50_without_a_note():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_sample_size_app, kwargs={"n_rows": 418}).run()
    assert app.session_state["n_used"] == 368
    assert not app.info
    assert "50 polymers not used" in app.get("download_button")[0].proto.label


@pytest.mark.parametrize("n_rows", [418, 7367])
def test_using_every_polymer_leaves_no_external_set(n_rows):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_sample_size_app, kwargs={"n_rows": n_rows}).run()
    app.number_input[0].set_value(n_rows).run()
    assert app.session_state["n_used"] == n_rows
    assert not app.get("download_button")
    assert "none are left" in app.caption[0].value


def test_number_of_external_polymers_can_be_chosen():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_sample_size_app, kwargs={"n_rows": 7367}).run()
    external_input = app.number_input[1]
    assert external_input.value == 100 and "of 6,367 left out" in external_input.label

    external_input.set_value(250).run()
    (button,) = app.get("download_button")
    assert "250 polymers not used" in button.proto.label and not button.proto.disabled
    assert not app.error


@pytest.mark.parametrize("n_external", [400, 0])
def test_out_of_range_external_number_disables_the_download(n_external):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_sample_size_app, kwargs={"n_rows": 7367}).run()
    app.number_input[0].set_value(7000).run()  # 367 left out
    app.number_input[1].set_value(n_external).run()

    assert "between 1 and 367" in app.error[0].value
    (button,) = app.get("download_button")
    assert button.proto.disabled and "400" not in button.proto.label
    assert app.session_state["n_used"] == 7000  # the experiment itself is unaffected


def test_growing_the_subset_never_offers_more_polymers_than_are_left():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_sample_size_app, kwargs={"n_rows": 7367}).run()
    app.number_input[1].set_value(250).run()
    app.number_input[0].set_value(7300).run()  # 67 left out

    # Depending on the Streamlit version, the number is either reset to its default
    # (the label changed) or kept and reported; never a download of 250.
    (button,) = app.get("download_button")
    if app.error:
        assert "between 1 and 67" in app.error[0].value and button.proto.disabled
    else:
        assert app.number_input[1].value <= 67
        assert f"{app.number_input[1].value} polymers not used" in button.proto.label


@pytest.mark.parametrize("n_used", [8000, 0])
def test_out_of_range_number_of_polymers_stops_the_form(n_used):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_sample_size_app, kwargs={"n_rows": 7367}).run()
    app.number_input[0].set_value(n_used).run()

    assert "between 1 and 7,367" in app.error[0].value
    assert "n_used" not in app.session_state  # nothing after the field ran
    assert not app.get("download_button")


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
