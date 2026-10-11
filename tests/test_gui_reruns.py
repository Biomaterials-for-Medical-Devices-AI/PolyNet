"""GUI reruns invalidate dependants, isolate analyses and warn before deletion."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from polynet.app.services import reruns
from polynet.app.services.explanation_results import capture_analysis, load_analysis
from polynet.config.enums import FragmentationMethod, ProblemType
from polynet.explainability import masking, shap_explain


def touch(path, value="old"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return path


@pytest.mark.parametrize("stage", ["representation", "training"])
def test_stage_cleanup_removes_all_dependants_and_preserves_inputs(tmp_path, stage):
    inputs = [
        touch(tmp_path / name)
        for name in [
            "data.csv",
            "data_options.json",
            "general_options.json",
            "config_used.yaml",
            "graph_feature_analysis.json",
            "notes.txt",
        ]
    ]
    outputs = [
        touch(tmp_path / name)
        for name in [
            "representation/Descriptors/rdkit.csv",
            "representation/GNN/processed/1.pt",
            "representation_options.json",
            "ml_results/models/GCN_1.pt",
            "ml_results/models/target_scaler_1.pkl",
            "ml_results/predictions.csv",
            "ml_results/metrics.json",
            "ml_results/plots/parity.png",
            "gnn_hyp_opt/iteration_1/search.csv",
            "tml_hyp_opt/rf.json",
            "hpo_search_spaces.json",
            "split_options.json",
            "split_indices.json",
            "train_gnn_options.json",
            "train_tml_options.json",
            "preprocessing_tml_options.json",
            "target_transform_options.json",
            "parity_plot.png",
            "explanations/analyses/old.json",
            "explanations/shap_rdkit.csv",
            "unseen_predictions/unseen/predictions.csv",
            "unseen_predictions/unseen/applicability_domain.json",
            "unseen_predictions/unseen/explanations/analyses/old.json",
        ]
    ]
    before = {p: p.read_bytes() for p in inputs}
    plan = reruns.stage_rerun_plan(tmp_path, stage)
    assert plan.has_results
    assert all(p.exists() for p in outputs)  # Planning never deletes.
    reruns.clear_stage_results(tmp_path, stage)
    assert all(p.read_bytes() == contents for p, contents in before.items())
    for path in outputs:
        if stage == "training" and path.parts[-1] in {
            "rdkit.csv",
            "1.pt",
            "representation_options.json",
        }:
            assert path.exists()
        else:
            assert not path.exists(), path
    assert not reruns.stage_rerun_plan(tmp_path, stage).has_results


def test_cleanup_unlinks_generated_symlink_without_deleting_its_target(tmp_path):
    root = tmp_path / "experiment"
    root.mkdir()
    target = touch(tmp_path / "outside" / "keep.txt")
    (root / "ml_results").symlink_to(target.parent, target_is_directory=True)
    reruns.clear_stage_results(root, "training")
    assert target.read_text() == "old"
    assert not (root / "ml_results").is_symlink()


def tml_results(root):
    rows = []
    for name in ["LinearRegression", "RandomForest"]:
        for iteration in ["1", "2"]:
            for sid in ["1", "2"]:
                for class_idx in ["0", "1"]:
                    rows.append(
                        dict(
                            model_type=name,
                            iteration=iteration,
                            sample_id=sid,
                            class_idx=class_idx,
                            base_value=5.0,
                            f=2.0,
                        )
                    )
    shap_explain._save_shap_cache(pd.DataFrame(rows), root, "rdkit")
    return dict(
        models={"LinearRegression-rdkit_1": object()},
        experiment_path=root,
        explain_sample_ids=["1"],
        descriptor_dfs={"rdkit": pd.DataFrame({"f": [3.0, 4.0]}, index=["1", "2"])},
        problem_type=ProblemType.Classification,
        target_class=1,
    )


def test_tml_rerun_replaces_only_selected_pairs_and_analysis(tmp_path):
    params = tml_results(tmp_path)
    chosen = capture_analysis("TML", "global", params)
    other_params = {**params, "explain_sample_ids": ["1", "2"]}
    other = capture_analysis("TML", "global", other_params)
    other_before = other.read_bytes()
    sibling = touch(tmp_path / "explanations" / "other_plot.png")
    cache_before = shap_explain._load_shap_cache(tmp_path, "rdkit")
    plan = reruns.explanation_rerun_plan("TML", params)
    assert plan.paths == (chosen,)
    reruns.clear_explanation_results("TML", params)
    assert not chosen.exists() and other.read_bytes() == other_before and sibling.exists()
    cache = shap_explain._load_shap_cache(tmp_path, "rdkit")
    selected = (
        (cache_before.model_type == "LinearRegression")
        & (cache_before.iteration == "1")
        & (cache_before.sample_id == "1")
        & (cache_before.class_idx == "1")
    )
    pd.testing.assert_frame_equal(cache, cache_before.loc[~selected].reset_index(drop=True))
    # The other analysis carries an independent copy of the original raw score.
    assert load_analysis(other)["arguments"]["cache_by_descriptor"]["rdkit"].shape[0] == 2
    assert not reruns.explanation_rerun_plan("TML", params).has_results


def test_tml_rerun_respects_each_models_split_and_external_source(tmp_path):
    external = tmp_path / "unseen_predictions" / "data"
    params = tml_results(external)
    tml_results(tmp_path)
    experiment_cache = (tmp_path / "explanations" / "shap_rdkit.csv").read_bytes()
    params.update(
        experiment_path=tmp_path,
        cache_root=external,
        models={"LinearRegression-rdkit_1": object(), "RandomForest-rdkit_2": object()},
        explain_sample_ids=["1", "2"],
        samples_per_model={"LinearRegression-rdkit_1": {"1"}, "RandomForest-rdkit_2": {"2"}},
    )
    chosen = capture_analysis("TML", "global", params)
    reruns.clear_explanation_results("TML", params)
    assert not chosen.exists()
    remaining = shap_explain._load_shap_cache(external, "rdkit")
    pairs = set(
        zip(remaining.model_type, remaining.iteration, remaining.sample_id, remaining.class_idx)
    )
    assert ("LinearRegression", "1", "1", "1") not in pairs
    assert ("RandomForest", "2", "2", "1") not in pairs
    assert ("LinearRegression", "1", "2", "1") in pairs and ("RandomForest", "2", "1", "1") in pairs
    assert (tmp_path / "explanations" / "shap_rdkit.csv").read_bytes() == experiment_cache


@pytest.mark.parametrize(
    "problem,class_idx,class_key",
    [
        (ProblemType.Regression, None, "regression"),
        (ProblemType.Classification, 0, "0"),
        (ProblemType.Classification, None, "regression"),
    ],
)
def test_gnn_rerun_preserves_other_classes_fragmentations_models_and_sources(
    tmp_path, problem, class_idx, class_key
):
    def entry():
        return {
            "chemistry_masking": {
                key: {"CO": {"brics": {"CO": [2.0]}, "retro": {"CO": [3.0]}}}
                for key in ["regression", "0", "1"]
            }
        }

    raw = {
        "GCN": {"1": {"1": entry(), "2": entry()}, "2": {"1": entry()}},
        masking.MASKING_CACHE_VERSION_KEY: masking.MASKING_CACHE_VERSION,
    }
    path = touch(tmp_path / "explanations" / "explanation.json", json.dumps(raw))
    params = dict(
        models={"GCN_1": object()},
        experiment_path=tmp_path,
        dataset=[SimpleNamespace(idx=1, mols=["CO"]), SimpleNamespace(idx=2, mols=["CO"])],
        explain_mols=[1],
        problem_type=problem,
        target_class=class_idx,
        fragmentation_approach=FragmentationMethod.BRICS,
    )
    chosen = capture_analysis("GNN", "global", params)
    other = capture_analysis("GNN", "local", {**params, "explain_mols": [2]})
    external = touch(
        tmp_path / "unseen_predictions" / "data" / "explanations" / "explanation.json",
        json.dumps(raw),
    )
    reruns.clear_explanation_results("GNN", params)
    assert not chosen.exists() and other.exists() and external.read_text() == json.dumps(raw)
    updated = json.loads(path.read_text())
    del raw["GCN"]["1"]["1"]["chemistry_masking"][class_key]["CO"]["brics"]
    assert updated == raw
    assert not reruns.explanation_rerun_plan("GNN", params).has_results


def rerun_app(root, stage):
    from pathlib import Path

    import streamlit as st

    from polynet.app.services.reruns import clear_stage_results, rerun_warning, stage_rerun_plan

    root = Path(root)
    if rerun_warning(stage_rerun_plan(root, stage), stage) and st.button("Run"):
        clear_stage_results(root, stage)
        (root / "ml_results").mkdir(exist_ok=True)
        (root / "ml_results" / "new_result.txt").write_text("new")
        st.success("Finished")


def test_warning_requires_acknowledgement_and_resets_for_new_results(tmp_path):
    result = touch(tmp_path / "ml_results" / "old_result.txt")
    app = AppTest.from_function(
        rerun_app, kwargs={"root": str(tmp_path), "stage": "training"}, default_timeout=30
    ).run()
    assert app.warning and not app.button and result.exists()
    app.checkbox[0].check().run()
    assert app.button and result.exists()
    app.button[0].click().run()
    assert not app.exception and not result.exists() and app.success
    app.run()
    assert not app.checkbox[0].value and not app.button


def explanation_app(root):
    from pathlib import Path

    import streamlit as st

    from polynet.app.services.reruns import clear_explanation_results, explanation_rerun_button
    from polynet.config.enums import ProblemType

    params = dict(
        models={"LinearRegression-rdkit_1": object()},
        experiment_path=Path(root),
        explain_sample_ids=["1"],
        problem_type=ProblemType.Classification,
        target_class=1,
    )
    if explanation_rerun_button("Run explanation", "TML", params, "test"):
        clear_explanation_results("TML", params)
        st.success("Ready to compute")


def test_explanation_warning_precedes_cache_replacement(tmp_path):
    params = tml_results(tmp_path)
    chosen = capture_analysis("TML", "global", params)
    before = (tmp_path / "explanations" / "shap_rdkit.csv").read_bytes()
    app = AppTest.from_function(
        explanation_app, kwargs={"root": str(tmp_path)}, default_timeout=30
    ).run()
    assert app.warning and app.button[0].disabled and chosen.exists()
    assert (tmp_path / "explanations" / "shap_rdkit.csv").read_bytes() == before
    app.checkbox[0].check().run()
    app.button[0].click().run()
    assert not app.exception and app.success and not chosen.exists()


def test_invalid_training_splits_leave_previous_files_untouched(tmp_path, monkeypatch):
    # Load the actual GUI function without executing the page's top-level widgets.
    import polynet.pipeline as pipeline

    old = touch(tmp_path / "ml_results" / "predictions.csv")
    config = touch(tmp_path / "split_options.json")
    import polynet.app.options.file_paths as paths

    monkeypatch.setattr(paths, "polynet_experiment_path", lambda **kw: tmp_path)
    monkeypatch.setattr(
        pipeline,
        "compute_data_splits",
        lambda **kw: (_ for _ in ()).throw(ValueError("cannot split")),
    )
    page_path = Path(__file__).parents[1] / "polynet/app/pages/3_Train_Models.py"

    def invalid_training_app(path, root):
        import ast
        from pathlib import Path

        import pandas as pd
        import streamlit as st

        from polynet.app.options.state_keys import GeneralConfigStateKeys
        from polynet.config.enums import SplitMethod, SplitType
        from polynet.config.schemas import DataConfig, GeneralConfig, RepresentationConfig

        source = ast.parse(Path(path).read_text())
        module = ast.Module(
            body=[
                n
                for n in source.body
                if isinstance(n, (ast.Import, ast.ImportFrom))
                or isinstance(n, ast.FunctionDef)
                and n.name == "train_models"
            ],
            type_ignores=[],
        )
        namespace = {}
        exec(compile(module, path, "exec"), namespace)
        st.session_state.update(
            {
                GeneralConfigStateKeys.SplitType: SplitType.TrainValTest,
                GeneralConfigStateKeys.SplitMethod: SplitMethod.Random,
                GeneralConfigStateKeys.TestSize: 0.2,
                GeneralConfigStateKeys.ValidationSize: 0.2,
            }
        )
        namespace["load_options"] = lambda **kw: GeneralConfig(name="experiment", random_seed=42)
        namespace["read_csv"] = lambda *args, **kw: pd.DataFrame(
            {"smiles": ["CC"], "target": [1.0]}
        )
        namespace["train_models"](
            "experiment",
            {},
            None,
            {},
            RepresentationConfig(
                smiles_merge_approach="no_merging", molecular_descriptors={"rdkit": ["MolWt"]}
            ),
            DataConfig(
                data_name="data.csv",
                data_path=str(Path(root) / "data.csv"),
                smiles_cols=["smiles"],
                target_variable_col="target",
                problem_type="regression",
                num_classes=1,
            ),
        )

    app = AppTest.from_function(
        invalid_training_app,
        kwargs={"path": str(page_path), "root": str(tmp_path)},
        default_timeout=30,
    ).run()
    assert not app.exception and app.error
    assert old.read_text() == "old" and config.read_text() == "old"


def test_replacement_preserves_unselected_shap_float_precision(tmp_path):
    params = tml_results(tmp_path)
    path = tmp_path / "explanations" / "shap_rdkit.csv"
    raw = pd.read_csv(path, dtype=str)
    other = (raw.model_type == "RandomForest") & (raw.iteration == "2")
    raw.loc[other, "f"] = "1.2345678901234567"
    raw.to_csv(path, index=False)
    reruns.clear_explanation_results("TML", params)
    after = pd.read_csv(path, dtype=str)
    assert set(after.loc[after.model_type == "RandomForest", "f"]) == {"2.0", "1.2345678901234567"}


def test_explanation_service_recomputes_selected_scores_before_saving(tmp_path, monkeypatch):
    from polynet.app.services import explain_tml

    params = tml_results(tmp_path)
    chosen = capture_analysis("TML", "global", params)
    other = capture_analysis("TML", "local", {**params, "explain_sample_ids": ["2"]})
    old_snapshot = other.read_bytes()
    calls = []

    def compute(**kwargs):
        cache = shap_explain._load_shap_cache(tmp_path, "rdkit")
        mask = (
            (cache.model_type == "LinearRegression")
            & (cache.iteration == "1")
            & (cache.sample_id == "1")
            & (cache.class_idx == "1")
        )
        assert not mask.any() and not chosen.exists()
        calls.append("computed")
        new = pd.DataFrame(
            [
                dict(
                    model_type="LinearRegression",
                    iteration="1",
                    sample_id="1",
                    class_idx="1",
                    base_value=5.0,
                    f=99.0,
                )
            ]
        )
        shap_explain._save_shap_cache(pd.concat([cache, new], ignore_index=True), tmp_path, "rdkit")
        return {}

    monkeypatch.setattr(explain_tml, "compute_global_shap_attribution", compute)
    monkeypatch.setattr(explain_tml, "display_tml_global", lambda result: None)

    # Execute the actual decorated GUI service in a Streamlit session.
    def service_app(root):
        from pathlib import Path

        import pandas as pd

        from polynet.app.services.explain_tml import explain_tml_global
        from polynet.config.enums import ProblemType

        explain_tml_global(
            models={"LinearRegression-rdkit_1": object()},
            descriptor_dfs={"rdkit": pd.DataFrame({"f": [3.0, 4.0]}, index=["1", "2"])},
            experiment_path=Path(root),
            explain_sample_ids=["1"],
            problem_type=ProblemType.Classification,
            target_class=1,
        )

    app = AppTest.from_function(
        service_app, kwargs={"root": str(tmp_path)}, default_timeout=30
    ).run()
    assert not app.exception and calls == ["computed"]
    assert chosen.exists() and other.read_bytes() == old_snapshot
    assert load_analysis(chosen)["arguments"]["cache_by_descriptor"]["rdkit"]["f"].tolist() == [
        99.0
    ]


def test_numeric_gnn_rerun_merges_without_losing_unrelated_cached_scores(tmp_path, monkeypatch):
    from polynet.explainability import explain

    raw = {
        "GCN": {
            "1": {
                "1": {
                    "chemistry_masking": {
                        "regression": {"CO": {"brics": {"CO": [2.0]}, "retro": {"CO": [3.0]}}},
                        "1": {"CO": {"brics": {"CO": [4.0]}}},
                    }
                }
            }
        },
        masking.MASKING_CACHE_VERSION_KEY: masking.MASKING_CACHE_VERSION,
    }
    path = touch(tmp_path / "explanations" / "explanation.json", json.dumps(raw))
    params = dict(
        models={"GCN_1": object()},
        experiment_path=tmp_path,
        explain_mols=[1],
        problem_type=ProblemType.Regression,
        target_class=None,
        fragmentation_approach=FragmentationMethod.BRICS,
    )
    reruns.clear_explanation_results("GNN", params)

    def calculate(**kwargs):
        return {
            "GCN": {
                "1": {1: {"chemistry_masking": {"regression": {"CO": {"brics": {"CO": [99.0]}}}}}}
            }
        }

    monkeypatch.setattr(explain, "calculate_masking_attributions", calculate)
    explain.compute_and_cache_masking(dataset=[SimpleNamespace(idx=1, mols=["CO"])], **params)
    updated = json.loads(path.read_text())
    raw["GCN"]["1"]["1"]["chemistry_masking"]["regression"]["CO"]["brics"]["CO"] = [99.0]
    assert updated == raw
    pairs = json.loads(path.read_text(), object_pairs_hook=lambda values: values)
    # No duplicate key "1" at the molecule level in the serialised JSON.
    molecule_pairs = dict(dict(pairs)["GCN"])["1"]
    assert [key for key, _ in molecule_pairs] == ["1"]
