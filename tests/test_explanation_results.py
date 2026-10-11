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
    analysis_label,
    analysis_summary,
    attribution_coverage,
    attribution_set_membership,
    capture_analysis,
    list_saved_analyses,
    load_analysis,
    local_prediction_breakdowns,
    select_attribution_coverage,
)
from polynet.config.constants import DataSet
from polynet.config.display_names import prettify_label
from polynet.config.enums import (
    ExplanationAggregation,
    FragmentationMethod,
    ImportanceNormalisationMethod,
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
                        base_value=sign * 5.0,
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
    assert not app.exception and not app.error
    assert len(app.color_picker) == 2
    app.radio[-1].set_value("bar").run()
    app.number_input[0].set_value(1).run()
    app.color_picker[0].set_value("#0099aa").run()
    assert not app.exception and not app.error
    app.radio[0].set_value("Individual molecules").run()
    assert not app.exception and not app.error
    _widget(app.multiselect, "Molecules to display").set_value(["1", "2"]).run()
    assert not app.exception and not app.error
    _widget(app.multiselect, "Models to include").set_value(["RandomForest-rdkit"]).run()
    assert not app.exception and not app.error
    app.radio[2].set_value(ExplanationAggregation.Separate).run()
    assert not app.exception and not app.error
    assert path.read_bytes() == before


def _widget(widgets, label):
    return next(widget for widget in widgets if widget.label == label)


@pytest.fixture
def gnn_selection(tmp_path):
    def entry(score):
        return {"chemistry_masking": {"regression": {"CO": {"brics": {"CO": [score]}}}}}

    names = ["GCN_1", "GCN_2", "GAT_1", "GAT_2"]
    raw = {
        "GCN": {"1": {"1": entry(2.0), "2": entry(4.0)}, "2": {"1": entry(6.0), "2": entry(8.0)}},
        "GAT": {
            "1": {"1": entry(10.0), "2": entry(20.0)},
            "2": {"1": entry(30.0), "2": entry(40.0)},
        },
        MASKING_CACHE_VERSION_KEY: MASKING_CACHE_VERSION,
    }
    cache = explanation_json_file_path(tmp_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(raw))
    breakdowns = {
        sid: {
            "true": float(sid),
            "rows": [
                {
                    "Model": prettify_label(name.rsplit("_", 1)[0]),
                    "Bootstrap": int(name.rsplit("_", 1)[1]),
                    "Set": (
                        DataSet.Training
                        if int(sid) == int(name.rsplit("_", 1)[1])
                        else DataSet.Test
                    ),
                    "Predicted": "1.5",
                }
                for name in names
            ],
        }
        for sid in ["1", "2"]
    }
    params = dict(
        models=dict.fromkeys(names),
        experiment_path=tmp_path,
        dataset=[SimpleNamespace(idx=i, mols=["CO"]) for i in [1, 2]],
        explain_mols=[1, 2],
        problem_type=ProblemType.Regression,
        fragmentation_approach=FragmentationMethod.BRICS,
        prediction_breakdowns=breakdowns,
    )
    path = capture_analysis("GNN", "global", params)
    return path, load_analysis(path)


def test_saved_set_membership_is_per_model_bootstrap(gnn_selection):
    _, record = gnn_selection
    membership = attribution_set_membership(record)
    assert membership["1"]["GCN_1"] == DataSet.Training
    assert membership["1"]["GCN_2"] == DataSet.Test
    assert select_attribution_coverage(record, ["GCN_1", "GCN_2"], DataSet.Test) == {
        "1": {"GCN_2"},
        "2": {"GCN_1"},
    }
    assert select_attribution_coverage(record, ["GAT_1"], DataSet.Training) == {"1": {"GAT_1"}}
    # Missing model combinations and unknown split metadata never create scores.
    assert select_attribution_coverage(record, ["GAT_3"]) == {}
    record["prediction_breakdowns"] = {}
    assert not attribution_set_membership(record)
    assert select_attribution_coverage(record, ["GCN_1"], DataSet.Test) == {}
    assert select_attribution_coverage(record, ["GCN_1"]) == {"1": {"GCN_1"}, "2": {"GCN_1"}}


def test_tml_set_filters_match_saved_descriptor_model_pairs(tmp_path, tml_cache):
    _, params = tml_cache
    params["prediction_breakdowns"] = {
        "rdkit": {
            sid: {
                "true": "positive",
                "rows": [
                    {
                        "Model": prettify_label(name.rsplit("_", 1)[0]),
                        "Bootstrap": 1,
                        "Set": (
                            DataSet.Test if name.startswith("RandomForest") else DataSet.Training
                        ),
                    }
                    for name in params["models"]
                ],
            }
            for sid in ["1", "2"]
        }
    }
    record = load_analysis(capture_analysis("TML", "global", params))
    # Predictions list all models, but only two model/sample pairs were explained.
    assert attribution_set_membership(record) == {
        "1": {"LinearRegression-rdkit_1": DataSet.Training},
        "2": {"RandomForest-rdkit_1": DataSet.Test},
    }
    assert select_attribution_coverage(record, list(params["models"]), DataSet.Test) == {
        "2": {"RandomForest-rdkit_1"}
    }


def _capture_global_plot(result, **kwargs):
    import streamlit as st

    st.session_state["plotted_scores"] = result.attribution_dict
    st.session_state["plotted_models"] = result.n_models
    st.session_state["plotted_molecules"] = result.n_mols


def _capture_local_plot(results, breakdowns, **kwargs):
    import streamlit as st

    st.session_state["plotted_local"] = [
        (
            str(result.mol_idx),
            result.model_label,
            result.attribution_df["Attribution (Y − Y_masked)"].tolist(),
        )
        for result in results
    ]
    st.session_state["plotted_breakdowns"] = breakdowns


def test_page6_gnn_architectures_bootstraps_sets_and_molecules(gnn_selection, monkeypatch):
    from polynet.app.components.forms import explanation_plots

    path, _ = gnn_selection
    before = path.read_bytes()
    monkeypatch.setattr(explain, "compute_and_cache_masking", _no_inference)
    monkeypatch.setattr(explanation_plots, "display_gnn_global", _capture_global_plot)
    monkeypatch.setattr(explanation_plots, "display_gnn_local", _capture_local_plot)
    app = AppTest.from_function(
        plots_app, kwargs={"root": str(path.parents[2])}, default_timeout=30
    ).run()
    app.checkbox(key="analyse_explanations_enabled").check().run()
    _widget(app.radio, "Normalisation").set_value(
        ImportanceNormalisationMethod.NoNormalisation
    ).run()
    assert not app.exception and not app.error
    assert sorted(app.session_state["plotted_scores"]["CO"]) == [2, 4, 6, 8, 10, 20, 30, 40]
    assert app.session_state["plotted_molecules"] == 2
    assert app.session_state["plotted_models"] == 4
    _widget(app.multiselect, "GNN architectures to include").set_value(["GCN"]).run()
    assert sorted(app.session_state["plotted_scores"]["CO"]) == [2, 4, 6, 8]
    _widget(app.multiselect, "GNN architectures to include").set_value(["GAT"]).run()
    assert sorted(app.session_state["plotted_scores"]["CO"]) == [10, 20, 30, 40]
    _widget(app.multiselect, "Bootstrap iterations to include").set_value([2]).run()
    assert sorted(app.session_state["plotted_scores"]["CO"]) == [30, 40]
    _widget(app.selectbox, "Explained set").set_value(DataSet.Test).run()
    assert app.session_state["plotted_scores"]["CO"] == [30]
    # Both bootstraps: molecule 1 is test in split 2, molecule 2 in split 1.
    _widget(app.multiselect, "Bootstrap iterations to include").set_value([1, 2]).run()
    assert sorted(app.session_state["plotted_scores"]["CO"]) == [20, 30]
    _widget(app.checkbox, "Manually refine molecule selection").check().run()
    _widget(app.multiselect, "Molecules to display").set_value(["2"]).run()
    assert app.session_state["plotted_scores"]["CO"] == [20]
    _widget(app.radio, "Plot view").set_value("Individual molecules").run()
    _widget(app.radio, "Normalisation").set_value(
        ImportanceNormalisationMethod.NoNormalisation
    ).run()
    _widget(app.multiselect, "Molecules to display").set_value(["1", "2"]).run()
    _widget(app.radio, "Models display").set_value(ExplanationAggregation.Separate).run()
    assert app.session_state["plotted_local"] == [
        ("1", f"{prettify_label('GAT')} — bootstrap 2", [30.0]),
        ("2", f"{prettify_label('GAT')} — bootstrap 1", [20.0]),
    ]
    for sid, bootstrap in [("1", 2), ("2", 1)]:
        rows = app.session_state["plotted_breakdowns"][sid]["rows"]
        assert (
            len(rows) == 1 and rows[0]["Bootstrap"] == bootstrap and rows[0]["Set"] == DataSet.Test
        )
    assert not app.exception and not app.error
    assert path.read_bytes() == before


@pytest.mark.parametrize("family", ["GNN", "TML"])
@pytest.mark.parametrize("aggregation", list(ExplanationAggregation))
def test_local_set_filter_preserves_population_normalisation(
    family, aggregation, gnn_selection, tml_cache
):
    if family == "GNN":
        _, record = gnn_selection
        arguments = record["arguments"]
        results = explain.render_cached_local_attribution(
            combined_explanations=arguments["combined_explanations"],
            model_names=["GCN_1", "GCN_2"],
            dataset=arguments["dataset"],
            explain_mols=[1, 2],
            problem_type=ProblemType.Regression,
            mols_per_model={"GCN_1": {"1"}, "GCN_2": {"2"}},
            aggregation=aggregation,
        )
        assert [
            result.attribution_df["Attribution (Y − Y_masked)"].iloc[0] for result in results
        ] == [2.0, 8.0]
        assert [result.attribution_df["Normalised Attribution"].iloc[0] for result in results] == [
            0.5,
            1.0,
        ]
    else:
        frame, params = tml_cache
        results = shap_explain.render_cached_local_shap_attribution(
            cache_by_descriptor={"rdkit": frame},
            model_names=list(params["models"]),
            descriptor_dfs=params["descriptor_dfs"],
            explain_sample_ids=["1", "2"],
            problem_type=ProblemType.Classification,
            target_class=1,
            samples_per_model={"LinearRegression-rdkit_1": {"1"}, "RandomForest-rdkit_1": {"2"}},
            aggregation=aggregation,
        )["rdkit"]
        assert [
            result.attribution_df.set_index("Feature").loc["f", "SHAP Value"] for result in results
        ] == [2.0, 4.0]
        assert [
            result.attribution_df.set_index("Feature").loc["f", "Normalised SHAP"]
            for result in results
        ] == [0.5, 1.0]
    plt.close("all")


def test_page6_exact_combinations_and_empty_selections(gnn_selection, monkeypatch):
    from polynet.app.components.forms import explanation_plots

    path, _ = gnn_selection
    monkeypatch.setattr(explanation_plots, "display_gnn_global", _capture_global_plot)
    monkeypatch.setattr(explain, "compute_and_cache_masking", _no_inference)
    app = AppTest.from_function(
        plots_app, kwargs={"root": str(path.parents[2])}, default_timeout=30
    ).run()
    app.checkbox(key="analyse_explanations_enabled").check().run()
    _widget(app.radio, "Normalisation").set_value(
        ImportanceNormalisationMethod.NoNormalisation
    ).run()
    _widget(app.checkbox, "Choose individual model/bootstrap combinations").check().run()
    _widget(app.multiselect, "Cached models / bootstraps to show").set_value(
        ["GAT_2", "GCN_1"]
    ).run()
    assert sorted(app.session_state["plotted_scores"]["CO"]) == [2, 4, 30, 40]
    _widget(app.multiselect, "GNN architectures to include").set_value(["GCN"]).run()
    assert _widget(app.multiselect, "Cached models / bootstraps to show").value == ["GCN_1"]
    assert sorted(app.session_state["plotted_scores"]["CO"]) == [2, 4]
    _widget(app.multiselect, "Bootstrap iterations to include").set_value([]).run()
    assert not app.exception and not app.error
    assert "Select at least one" in app.info[0].value


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


def test_views_share_one_snapshot_and_simple_label(tmp_path, tml_cache):
    _, params = tml_cache
    path = capture_analysis("TML", "global", params)
    assert capture_analysis("TML", "local", params) == path
    assert len(list_saved_analyses(tmp_path)) == 1
    record = load_analysis(path)
    assert attribution_coverage(record) == {
        "1": {"LinearRegression-rdkit_1"},
        "2": {"RandomForest-rdkit_1"},
    }
    label = analysis_label(analysis_summary(path))
    assert "2026" not in label and "global" not in label and "local" not in label
    assert "2 samples" in label and "class 1" in label


def test_prediction_breakdown_matches_cached_and_selected_combinations(tmp_path, tml_cache):
    _, params = tml_cache
    params["samples_per_model"] = None
    params["prediction_breakdowns"] = {
        "rdkit": {
            "1": {
                "true": "positive",
                "rows": [
                    {"Model": "Linear Regression RDKit", "Bootstrap": 1, "Predicted": "positive"},
                    {"Model": "Random Forest RDKit", "Bootstrap": 1, "Predicted": "negative"},
                ],
            }
        }
    }
    record = load_analysis(capture_analysis("TML", "global", params))
    from polynet.config.display_names import prettify_label

    rows = record["prediction_breakdowns"]["rdkit"]["1"]["rows"]
    rows[0]["Model"] = prettify_label("LinearRegression-rdkit")
    rows[1]["Model"] = prettify_label("RandomForest-rdkit")
    selected = local_prediction_breakdowns(record, ["1"], ["RandomForest-rdkit_1"])
    assert selected["rdkit"]["1"]["rows"] == [rows[1]]


@pytest.mark.parametrize(
    "kind,class_idx",
    [
        ("linear", None),
        ("forest", None),
        ("logistic", 0),
        ("logistic", 1),
        ("classifier", 0),
        ("classifier", 1),
    ],
)
def test_cached_baseline_reconstructs_fitted_model_output(tmp_path, monkeypatch, kind, class_idx):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegression

    x = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 1.0], [3.0, 2.0], [4.0, 1.0], [5.0, 3.0]])
    y = np.array([0, 0, 0, 1, 1, 1]) if class_idx is not None else 7 + x[:, 0] * 3 - x[:, 1]
    model = {
        "linear": LinearRegression,
        "forest": lambda: RandomForestRegressor(n_estimators=5, random_state=3),
        "logistic": LogisticRegression,
        "classifier": lambda: RandomForestClassifier(n_estimators=5, random_state=3),
    }[kind]().fit(x, y)
    problem = ProblemType.Regression if class_idx is None else ProblemType.Classification
    transformer = SimpleNamespace(
        get_feature_names_out=lambda: ["f", "g"], transform=lambda df: df.to_numpy()
    )
    monkeypatch.setattr(shap_explain, "_load_feature_transformer", lambda *args: transformer)
    name = f"{type(model).__name__}-rdkit_1"
    cache = shap_explain.compute_and_cache_shap(
        models={name: model},
        descriptor_dfs={
            "rdkit": pd.DataFrame(x, columns=["f", "g"], index=list(map(str, range(len(x)))))
        },
        experiment_path=tmp_path,
        problem_type=problem,
        explain_sample_ids=["0", "4"],
        target_class=class_idx,
    )["rdkit"].set_index("sample_id")
    actual = cache[["f", "g"]].sum(axis=1) + cache["base_value"]
    inputs = x[[int(sid) for sid in cache.index]]
    if kind == "logistic":
        expected = model.decision_function(inputs) * (-1 if class_idx == 0 else 1)
    elif kind == "classifier":
        expected = model.predict_proba(inputs)[:, class_idx]
    else:
        expected = model.predict(inputs)
    np.testing.assert_allclose(actual.to_numpy(), expected, atol=1e-7)
    cached = shap_explain._load_shap_cache(tmp_path, "rdkit")
    assert "base_value" in cached
    pd.testing.assert_frame_equal(
        cache.reset_index().sort_values("sample_id").reset_index(drop=True),
        cached[cache.reset_index().columns].sort_values("sample_id").reset_index(drop=True),
    )


@pytest.mark.parametrize("aggregation", list(ExplanationAggregation))
def test_local_shap_permodel_uses_population_max_and_scales_baseline(monkeypatch, aggregation):
    frame = pd.DataFrame(
        [
            dict(
                model_type="LinearRegression",
                iteration="1",
                sample_id="1",
                class_idx="regression",
                f=2.0,
                g=1.0,
                base_value=8.0,
            ),
            dict(
                model_type="LinearRegression",
                iteration="1",
                sample_id="2",
                class_idx="regression",
                f=-4.0,
                g=1.0,
                base_value=8.0,
            ),
            dict(
                model_type="LinearRegression",
                iteration="2",
                sample_id="1",
                class_idx="regression",
                f=10.0,
                g=0.0,
                base_value=20.0,
            ),
            dict(
                model_type="LinearRegression",
                iteration="2",
                sample_id="2",
                class_idx="regression",
                f=100.0,
                g=0.0,
                base_value=20.0,
            ),
        ]
    )
    before = frame.copy(deep=True)
    plots = []

    def plot(**kwargs):
        plots.append(kwargs)
        return plt.figure()

    monkeypatch.setattr(shap_explain, "plot_local_shap", plot)
    result = shap_explain.render_cached_local_shap_attribution(
        cache_by_descriptor={"rdkit": frame},
        model_names=["LinearRegression-rdkit_1", "LinearRegression-rdkit_2"],
        descriptor_dfs={},
        explain_sample_ids=["1"],
        problem_type=ProblemType.Regression,
        aggregation=aggregation,
    )["rdkit"]
    if aggregation == ExplanationAggregation.Average:
        np.testing.assert_allclose(plots[0]["values"], [0.3, 0.125])
        assert plots[0]["base_value"] == pytest.approx(1.1)
        assert result[0].attribution_df["SHAP Value"].tolist() == [6.0, 0.5]
    else:
        np.testing.assert_allclose(plots[0]["values"], [0.5, 0.25])
        np.testing.assert_allclose(plots[1]["values"], [0.1, 0])
        assert [p["base_value"] for p in plots] == [2.0, 0.2]
    pd.testing.assert_frame_equal(frame, before)


def test_model_selected_features_contribute_zero_to_ensemble(monkeypatch):
    frame = pd.DataFrame(
        [
            dict(
                model_type="LinearRegression",
                iteration="1",
                sample_id="1",
                class_idx="regression",
                f=2.0,
                g=np.nan,
                base_value=3.0,
            ),
            dict(
                model_type="RandomForest",
                iteration="1",
                sample_id="1",
                class_idx="regression",
                f=np.nan,
                g=8.0,
                base_value=5.0,
            ),
        ]
    )
    plots = []

    def plot(**kwargs):
        plots.append(kwargs)
        return plt.figure()

    monkeypatch.setattr(shap_explain, "plot_local_shap", plot)
    args = dict(
        cache_by_descriptor={"rdkit": frame},
        model_names=["LinearRegression-rdkit_1", "RandomForest-rdkit_1"],
        descriptor_dfs={},
        explain_sample_ids=["1"],
        problem_type=ProblemType.Regression,
        normalisation_type=ImportanceNormalisationMethod.NoNormalisation,
    )
    shap_explain.render_cached_local_shap_attribution(**args)
    assert plots[0]["base_value"] + sum(plots[0]["values"]) == 9.0  # mean(3+2, 5+8)
    merged = shap_explain.merge_shap_attributions(frame, args["model_names"], ["1"])
    assert merged.loc["1"].tolist() == [1.0, 4.0]


def test_force_keeps_all_contributions_and_baseline(monkeypatch):
    import shap

    captured = []

    def force(explanation, **kwargs):
        captured.append(explanation)
        plt.figure()

    monkeypatch.setattr(shap.plots, "force", force)
    fig = shap_explain.plot_local_shap(
        values=np.array([2.0, 3.0, -1.0]),
        feature_names=["f", "g", "h"],
        base_value=7.0,
        max_display=1,
        plot_type="force",
    )
    assert captured[0].base_values == 7.0
    assert captured[0].values.tolist() == [2.0, 3.0, -1.0]
    plt.close(fig)


@pytest.mark.parametrize("aggregation", list(ExplanationAggregation))
def test_gnn_local_permodel_uses_saved_population_and_available_pairs(monkeypatch, aggregation):
    def entry(score):
        return {"chemistry_masking": {"regression": {"CO": {"brics": {"CO": [score]}}}}}

    raw = {
        "GCN": {
            "1": {"1": entry(2.0), "2": entry(-4.0)},
            "2": {"1": entry(10.0), "2": entry(100.0)},
            "3": {"2": entry(30.0)},
        }
    }
    before = json.dumps(raw)
    results = explain.render_cached_local_attribution(
        combined_explanations=raw,
        model_names=["GCN_1", "GCN_2", "GCN_3"],
        dataset=[SimpleNamespace(idx=1, mols=["CO"])],
        explain_mols=[1],
        problem_type=ProblemType.Regression,
        aggregation=aggregation,
    )
    if aggregation == ExplanationAggregation.Average:
        assert results[0].attribution_df["Normalised Attribution"].tolist() == pytest.approx([0.3])
        assert results[0].attribution_df["Attribution (Y − Y_masked)"].tolist() == [6.0]
    else:
        assert len(results) == 2
        assert [
            r.attribution_df["Normalised Attribution"].iloc[0] for r in results
        ] == pytest.approx([0.5, 0.1])
    assert json.dumps(raw) == before
    plt.close("all")
