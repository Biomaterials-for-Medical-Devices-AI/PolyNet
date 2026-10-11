"""Saved explanation plotting: replay without inference, class/split isolation, UI."""

import inspect
import json
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from polynet.app.services.explanation_results import (
    capture_analysis,
    list_cached_analyses,
    list_saved_analyses,
    load_analysis,
    load_cached_analysis,
)
from polynet.config.enums import (
    ExplanationAggregation,
    FragmentationMethod,
    ImportanceNormalisationMethod,
    IteratorType,
    ProblemType,
)
from polynet.config.paths import explanation_json_file_path
from polynet.explainability import explain, masking, shap_explain
from polynet.explainability.masking import MASKING_CACHE_VERSION, MASKING_CACHE_VERSION_KEY


@pytest.fixture
def tml_cache(tmp_path):
    rows = []
    for model in ["LinearRegression", "RandomForest"]:
        for sid, value in [("1", 2.0), ("2", 4.0)]:
            for class_key, sign in [("0", -1), ("1", 1)]:
                rows.append(
                    dict(
                        model_type=model,
                        iteration="1",
                        sample_id=sid,
                        class_idx=class_key,
                        f=sign * value,
                        g=sign * value / 2,
                    )
                )
    frame = pd.DataFrame(rows)
    shap_explain._save_shap_cache(frame, tmp_path, "rdkit")
    descriptors = pd.DataFrame(
        {"f": [10.0, 20.0], "g": [3.0, 6.0], "target": [1, 2]}, index=["1", "2"]
    )
    params = dict(
        models={"LinearRegression-rdkit_1": object(), "RandomForest-rdkit_1": object()},
        descriptor_dfs={"rdkit": descriptors},
        experiment_path=tmp_path,
        explain_sample_ids=["1", "2"],
        problem_type=ProblemType.Classification,
        target_class=1,
        target_col="target",
        samples_per_model={"LinearRegression-rdkit_1": {"1"}, "RandomForest-rdkit_1": {"2"}},
        prediction_breakdowns={"rdkit": {"1": {"true": "positive", "rows": []}}},
    )
    return frame, params


def _render(record, renderer, **settings):
    params = inspect.signature(renderer).parameters
    arguments = {k: v for k, v in record["arguments"].items() if k in params}
    return renderer(**arguments, **settings)


def _no_inference(*args, **kwargs):
    pytest.fail("Plot customisation tried to compute missing attributions")


def test_tml_snapshot_preserves_class_split_features_and_external_root(
    tmp_path, tml_cache, monkeypatch
):
    frame, params = tml_cache
    external = tmp_path / "unseen_predictions" / "external"
    shap_explain._save_shap_cache(frame, external, "rdkit")
    params["cache_root"] = external
    path = capture_analysis("TML", "global", params)
    assert path.parent == external / "explanations" / "analyses"
    assert not list_saved_analyses(tmp_path)
    record = load_analysis(path)
    raw = record["arguments"]["cache_by_descriptor"]["rdkit"]
    assert set(raw["class_idx"]) == {"1"}
    assert list(zip(raw["model_type"], raw["sample_id"])) == [
        ("LinearRegression", "1"),
        ("RandomForest", "2"),
    ]
    pd.testing.assert_frame_equal(
        record["arguments"]["descriptor_dfs"]["rdkit"],
        params["descriptor_dfs"]["rdkit"][["f", "g"]],
        check_dtype=False,
    )
    monkeypatch.setattr(shap_explain, "compute_and_cache_shap", _no_inference)
    for style in ["beeswarm", "bar", "violin"]:
        result = _render(
            record,
            shap_explain.render_cached_global_shap_attribution,
            normalisation_type=ImportanceNormalisationMethod.NoNormalisation,
            plot_type=style,
        )["rdkit"]
        assert result.attribution_dict["f"] == [2.0, 4.0]
        assert result.figure is not None
        plt.close("all")
    # Plotting does not rewrite raw values or metadata.
    assert load_analysis(path)["arguments"]["cache_by_descriptor"]["rdkit"].equals(raw)


@pytest.mark.parametrize("aggregation", list(ExplanationAggregation))
def test_render_shap_filters_class_and_selected_models(
    tmp_path, tml_cache, monkeypatch, aggregation
):
    frame, params = tml_cache
    monkeypatch.setattr(shap_explain, "compute_and_cache_shap", _no_inference)
    for class_idx, sign in [(0, -1), (1, 1)]:
        record = {
            "arguments": dict(
                cache_by_descriptor={"rdkit": frame},
                model_names=["LinearRegression-rdkit_1"],
                descriptor_dfs=params["descriptor_dfs"],
                explain_sample_ids=["1"],
                problem_type=ProblemType.Classification,
                target_class=class_idx,
            )
        }
        for style in ["waterfall", "force", "bar"]:
            result = _render(
                record,
                shap_explain.render_cached_local_shap_attribution,
                aggregation=aggregation,
                local_plot_type=style,
                normalisation_type=ImportanceNormalisationMethod.NoNormalisation,
            )["rdkit"][0]
            assert result.attribution_df["SHAP Value"].tolist() == [sign * 2.0, sign]
            assert result.figure is not None and not result.warning
            plt.close("all")


def _gnn_cache(root):
    raw = {
        "GCN": {
            "1": {"1": {"chemistry_masking": {"regression": {"CO": {"brics": {"CO": [2.0]}}}}}}
        },
        MASKING_CACHE_VERSION_KEY: MASKING_CACHE_VERSION,
    }
    path = explanation_json_file_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw))
    return raw


def test_gnn_local_snapshot_keeps_numeric_ids_labels_names_and_aggregation(tmp_path, monkeypatch):
    _gnn_cache(tmp_path)
    params = dict(
        models={"GCN_1": object()},
        experiment_path=tmp_path,
        dataset=[SimpleNamespace(idx=1, mols=["CO"])],
        explain_mols=[1],
        problem_type=ProblemType.Regression,
        fragmentation_approach=FragmentationMethod.BRICS,
        predictions={1: {"True": np.nan, "Predicted": np.float64(2.1)}},
        prediction_breakdowns={"1": {"true": 2.0, "rows": [{"Model": "GCN", "Predicted": 2.1}]}},
    )
    path = capture_analysis("GNN", "local", params)
    record = load_analysis(path)
    assert record["arguments"]["dataset"][0].idx == 1
    assert record["prediction_breakdowns"] == params["prediction_breakdowns"]
    monkeypatch.setattr(explain, "compute_and_cache_masking", _no_inference)
    for aggregation in ExplanationAggregation:
        result = _render(
            record,
            explain.render_cached_local_attribution,
            aggregation=aggregation,
            mol_names={1: ["Custom name"]},
        )[0]
        assert not result.warning
        assert result.attribution_df["Attribution (Y − Y_masked)"].tolist() == [2.0]
        assert result.predicted_label == "2.1"
        assert result.mol_figure is not None
        plt.close("all")


def test_legacy_caches_can_be_loaded_without_models(tmp_path, tml_cache, monkeypatch):
    _gnn_cache(tmp_path)
    entries = list_cached_analyses(tmp_path)
    assert {entry["family"] for entry in entries} == {"TML", "GNN"}
    monkeypatch.setattr(explain, "compute_and_cache_masking", _no_inference)
    monkeypatch.setattr(shap_explain, "compute_and_cache_shap", _no_inference)
    for entry in entries:
        problem = (
            ProblemType.Regression
            if entry["class_key"] == "regression"
            else ProblemType.Classification
        )
        options = SimpleNamespace(problem_type=problem, target_variable_col="target")
        for scope in ["global", "local"]:
            record = load_cached_analysis(entry, tmp_path, scope, options)
            renderer = {
                ("GNN", "global"): explain.render_cached_global_attribution,
                ("GNN", "local"): explain.render_cached_local_attribution,
                ("TML", "global"): shap_explain.render_cached_global_shap_attribution,
                ("TML", "local"): shap_explain.render_cached_local_shap_attribution,
            }[entry["family"], scope]
            assert _render(record, renderer)
            plt.close("all")


def plots_app(root):
    from pathlib import Path
    from types import SimpleNamespace

    from polynet.app.components.forms.explanation_plots import explanation_plots_form
    from polynet.config.enums import ProblemType

    explanation_plots_form(Path(root), SimpleNamespace(problem_type=ProblemType.Classification))


def test_page6_load_and_customise_never_runs_inference(tmp_path, tml_cache, monkeypatch):
    _, params = tml_cache
    path = capture_analysis("TML", "global", params)
    before = path.read_bytes()
    monkeypatch.setattr(shap_explain, "compute_and_cache_shap", _no_inference)
    app = AppTest.from_function(plots_app, kwargs={"root": str(tmp_path)}, default_timeout=30).run()
    app.checkbox(key="analyse_explanations_enabled").check().run()
    assert not app.exception
    app.button(key="analyse_explanation_load").click().run()
    assert not app.exception and not app.error
    assert len(app.color_picker) == 2
    app.radio[-1].set_value("bar").run()
    app.number_input[0].set_value(1).run()
    app.color_picker[0].set_value("#0099aa").run()
    assert not app.exception and not app.error
    assert path.read_bytes() == before


def test_no_completed_analysis_shows_instruction(tmp_path):
    app = AppTest.from_function(plots_app, kwargs={"root": str(tmp_path)}).run()
    assert not app.exception and not app.button
    assert "Explain Models first" in app.info[0].value


def tml_selection_app():
    from types import SimpleNamespace

    import streamlit as st

    from polynet.app.components.forms.explain_tml import _tml_shared_params_section
    from polynet.config.enums import ProblemType

    st.session_state["params"] = _tml_shared_params_section(
        ["RandomForest-rdkit_1"],
        {},
        SimpleNamespace(problem_type=ProblemType.Classification, class_names=None, num_classes=2),
    )


def test_page5_numeric_target_class_and_default_presentation():
    app = AppTest.from_function(tml_selection_app).run()
    app.toggle[0].set_value(True).run()
    app.selectbox[0].set_value(1).run()
    assert not app.exception
    assert app.session_state["params"]["target_class"] == 1
    assert (
        app.session_state["params"]["normalisation_type"] == ImportanceNormalisationMethod.PerModel
    )
    assert not app.color_picker and not app.radio


@pytest.mark.parametrize("target_class, sign", [(0, -1), (1, 1)])
def test_global_shap_ignores_other_cached_classes(tml_cache, monkeypatch, target_class, sign):
    frame, params = tml_cache
    monkeypatch.setattr(shap_explain, "compute_and_cache_shap", _no_inference)
    result = shap_explain.render_cached_global_shap_attribution(
        cache_by_descriptor={"rdkit": frame},
        model_names=list(params["models"]),
        descriptor_dfs=params["descriptor_dfs"],
        explain_sample_ids=["1", "2"],
        problem_type=ProblemType.Classification,
        target_class=target_class,
        normalisation_type=ImportanceNormalisationMethod.NoNormalisation,
        plot_type="bar",
    )["rdkit"]
    assert result.attribution_dict["f"] == [sign * 2.0, sign * 4.0]
    plt.close("all")


def test_snapshot_version_is_checked(tmp_path, tml_cache):
    _, params = tml_cache
    path = capture_analysis("TML", "global", params)
    record = json.loads(path.read_text())
    record["attribution_version"] = -1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="older version"):
        load_analysis(path)


def test_snapshot_float_values_round_trip_exactly(tmp_path, tml_cache):
    frame, params = tml_cache
    frame.loc[0, "f"] = 1.2345678901234567
    frame["class_idx"] = "1"
    shap_explain._save_shap_cache(frame, tmp_path, "rdkit")
    expected = shap_explain._load_shap_cache(tmp_path, "rdkit")
    params["samples_per_model"] = None
    actual = load_analysis(capture_analysis("TML", "global", params))["arguments"][
        "cache_by_descriptor"
    ]["rdkit"]
    assert actual["f"].tolist() == expected["f"].tolist()


def test_legacy_discovery_refreshes_after_cache_rewrite(tmp_path, tml_cache):
    entries = list_cached_analyses(tmp_path)
    assert len(entries) == 2
    # Altering a returned cached frame must not pollute the next caller.
    entries[0]["raw"]["rdkit"].iloc[0, -1] = 999
    assert list_cached_analyses(tmp_path)[0]["raw"]["rdkit"].iloc[0, -1] != 999
    frame, _ = tml_cache
    frame["class_idx"] = "regression"
    shap_explain._save_shap_cache(frame, tmp_path, "rdkit")
    updated = list_cached_analyses(tmp_path)
    assert len(updated) == 1 and updated[0]["class_key"] == "regression"


def test_numeric_gnn_ids_reuse_existing_attributions(tmp_path, monkeypatch):
    raw = _gnn_cache(tmp_path)
    monkeypatch.setattr(masking, "_compute_masking_attributions", _no_inference)
    result = masking.calculate_masking_attributions(
        mols=[SimpleNamespace(idx=1, mols=["CO"])],
        models={"GCN_1": SimpleNamespace(eval=lambda: None)},
        fragmentation_method=FragmentationMethod.BRICS,
        problem_type=ProblemType.Regression,
        existing_explanations=raw,
    )
    assert result["GCN"]["1"][1]["chemistry_masking"]["regression"]["CO"]["brics"]["CO"] == [2.0]


def test_legacy_local_panels_load_stored_prediction_breakdowns(tmp_path, tml_cache):
    _gnn_cache(tmp_path)
    path = tmp_path / "ml_results" / "predictions.csv"
    path.parent.mkdir()
    pd.DataFrame(
        {
            IteratorType.BootstrapIteration.value: [1, 1],
            "True target": [0, 1],
            "SET": ["train", "test"],
            "LinearRegression-rdkit Predicted target": [0, 1],
            "RandomForest-rdkit Predicted target": [0, 1],
            "GCN Predicted target": [0, 1],
        },
        index=["1", "2"],
    ).to_csv(path)
    for entry in list_cached_analyses(tmp_path):
        options = SimpleNamespace(
            problem_type=(
                ProblemType.Regression if entry["family"] == "GNN" else ProblemType.Classification
            ),
            target_variable_col="target",
            target_variable_name="target",
            class_names=None,
        )
        record = load_cached_analysis(entry, tmp_path, "local", options)
        breakdown = record["prediction_breakdowns"]
        if entry["family"] == "TML":
            breakdown = breakdown["rdkit"]
        assert breakdown["1"]["rows"][0]["Predicted"] == "0"
        assert breakdown["1"]["true"] == "0"
