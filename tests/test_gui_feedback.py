"""GUI safeguards: defaults, stale caches, session isolation and workload limits."""

import logging
import threading

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from polynet.app.services.cache import read_csv, read_json
from polynet.app.services.progress import RunLogHandler
from polynet.app.services.workload import estimate_workload


def test_file_cache_refresh_and_copy(tmp_path, monkeypatch):
    path = tmp_path / "data.csv"
    path.write_text("x\n1\n")
    calls = []
    original = pd.read_csv

    def tracked(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", tracked)
    frame = read_csv(path)
    frame.iloc[0, 0] = 99
    assert read_csv(path).iloc[0, 0] == 1
    assert len(calls) == 1
    path.write_text("x\n2\n")  # same length still invalidates
    assert read_csv(path).iloc[0, 0] == 2
    path.unlink()
    with pytest.raises(FileNotFoundError):
        read_csv(path)
    path.write_text("x\n3\n")
    assert read_csv(path).iloc[0, 0] == 3
    other = tmp_path / "other.csv"
    other.write_text("x\n4\n")
    assert read_csv(other).iloc[0, 0] == 4
    options = tmp_path / "options.json"
    options.write_text('{"values": [1]}')
    read_json(options)["values"].append(2)
    assert read_json(options) == {"values": [1]}
    options.write_text('{"values": [3]}')
    assert read_json(options) == {"values": [3]}


def test_workload_thresholds_and_multipliers():
    args = dict(rows=100, representations=1, tml_models=1, gnn_models=0, repetitions=1)
    assert not estimate_workload(**args).reasons
    assert not estimate_workload(**{**args, "repetitions": 9}).reasons
    assert estimate_workload(**{**args, "repetitions": 10}).reasons
    tuned = estimate_workload(**args, tml_samples=10, tml_folds=5)
    assert tuned.final_fits == 1 and tuned.fit_budget == 51 and tuned.reasons
    assert not estimate_workload(**{**args, "rows": 5000}).reasons
    assert estimate_workload(**{**args, "rows": 5000, "tml_models": 2}).reasons
    assert estimate_workload(**{**args, "rows": 5000, "tml_models": 0, "gnn_models": 1}).reasons
    hybrid = estimate_workload(
        **{**args, "representations": 3, "tml_models": 2, "gnn_models": 2, "repetitions": 2},
        gnn_samples=3,
    )
    assert hybrid.final_fits == 16 and hybrid.fit_budget == 76
    assert hybrid.gnn_epochs == 16000


def test_log_tail_is_bounded_and_thread_scoped():
    rendered = []
    handler = RunLogHandler(rendered.append)
    record = lambda message: logging.LogRecord(
        "polynet.training", logging.INFO, "", 1, message, (), None
    )
    thread = threading.Thread(target=lambda: handler.handle(record("another session")))
    thread.start()
    thread.join()
    assert not rendered
    for i in range(100):
        handler.handle(record(f"event {i}"))
    assert len(handler.lines) == 80
    assert "another session" not in rendered[-1]
    assert rendered[-1].endswith("event 99")


def training_form_app(problem="regression"):
    import streamlit as st

    from polynet.app.components.forms.train_models import (
        split_data_form,
        train_GNN_models_form,
        train_TML_models,
    )
    from polynet.config.enums import ProblemType
    from polynet.config.schemas import RepresentationConfig

    advanced = st.toggle("Advanced", key="advanced")
    problem = ProblemType(problem)
    st.session_state["tml"] = train_TML_models(problem, advanced=advanced)
    st.session_state["gnn"] = train_GNN_models_form(
        RepresentationConfig(
            smiles_merge_approach="no_merging", molecular_descriptors={"rdkit": ["MolWt"]}
        ),
        problem,
        advanced=advanced,
    )
    split_data_form(problem, advanced=advanced)


@pytest.mark.parametrize("problem", ["regression", "classification"])
def test_training_defaults_match_unlocked_and_relock_resets(problem):
    app = AppTest.from_function(
        training_form_app, kwargs={"problem": problem}, default_timeout=30
    ).run()
    app.toggle(key="TrainTML").set_value(True)
    app.toggle(key="TrainGNN").set_value(True).run()
    for key in [
        "LinearRegression" if problem == "regression" else "LogisticRegression",
        "RandomForest",
        "SupportVectorMachine",
        "XGBoost",
    ]:
        app.toggle(key=key).set_value(True)
    app.run()
    assert not app.exception
    tml = app.session_state["tml"]
    gnn = app.session_state["gnn"]
    assert tml and all(tml.values())
    assert gnn and all(gnn.values())
    app.toggle(key="advanced").set_value(True).run()
    assert not app.exception
    assert app.session_state["tml"] == tml
    assert app.session_state["gnn"] == gnn
    app.checkbox(key="HypTunning").set_value(True).run()
    assert not app.exception
    assert not any(app.session_state["gnn"].values())
    app.select_slider(key="BootstrapIterations").set_value(5).run()
    app.toggle(key="advanced").set_value(False).run()
    assert not app.exception
    assert app.session_state["tml"] == tml
    assert app.session_state["gnn"] == gnn
    assert app.session_state["BootstrapIterations"] == 1
    assert app.session_state["HypTunning"] is False


def representation_form_app(string_representation="smiles"):
    import pandas as pd
    import streamlit as st

    from polynet.app.components.forms.representation import (
        basic_representation,
        cached_graph_feature_analysis,
        graph_representation,
        molecular_descriptor_representation,
    )
    from polynet.config.schemas import DataConfig

    advanced = st.toggle("Advanced")
    df = pd.DataFrame(
        {
            "smiles": (["*CC*", "*CCO*"] if string_representation == "psmiles" else ["CC", "CCO"]),
            "target": [1, 2],
        }
    )
    cfg = DataConfig(
        data_name="data.csv",
        data_path="data.csv",
        id_col="id",
        smiles_cols=["smiles"],
        target_variable_col="target",
        target_variable_name="Target",
        problem_type="regression",
        string_representation=string_representation,
        num_classes=1,
    )
    analysis = cached_graph_feature_analysis(df, cfg.smiles_cols)
    if advanced:
        descriptors = molecular_descriptor_representation(df, cfg)
        nodes, edges = graph_representation(cfg, df, analysis)
    else:
        descriptors, nodes, edges = basic_representation(cfg, df, analysis)
    st.session_state["config"] = (descriptors, nodes, edges)


@pytest.mark.parametrize("string_representation", ["smiles", "psmiles"])
def test_representation_defaults_match_unlocked(string_representation):
    app = AppTest.from_function(
        representation_form_app,
        kwargs={"string_representation": string_representation},
        default_timeout=30,
    ).run()
    assert not app.exception
    config = app.session_state["config"]
    app.toggle[0].set_value(True).run()
    assert not app.exception
    assert app.session_state["config"] == config


def test_progress_cleanup_after_failure():
    def app_script():
        import logging

        import streamlit as st

        from polynet.app.services.progress import run_progress

        logger = logging.getLogger("polynet")
        before = (logger.level, list(logger.handlers))
        try:
            with run_progress("Test run"):
                logger.info("GCN split/bootstrap 1/2")
                raise ValueError("expected")
        except ValueError:
            pass
        st.session_state["clean"] = before == (logger.level, list(logger.handlers))

    app = AppTest.from_function(app_script).run()
    assert not app.exception
    assert app.session_state["clean"]
    assert app.status[0].state == "error"
    assert "GCN split/bootstrap 1/2" in app.code[0].value


def test_ray_progress_reports_context(caplog):
    from types import SimpleNamespace

    from polynet.training.hyperopt import TrainingProgressCallback

    trial = SimpleNamespace(trial_id="trial-7")
    callback = TrainingProgressCallback("GCN", 2, 3)
    with caplog.at_level(logging.INFO):
        callback.on_trial_start(0, [], trial)
        callback.on_trial_result(0, [], trial, {"training_iteration": 1, "val_loss": 0.5})
        callback.on_trial_complete(0, [], trial)
        callback.on_trial_error(0, [], trial)
    assert "HPO GCN | split/bootstrap 2 | trial trial-7" in caplog.text
    assert "completed (1/3)" in caplog.text
    assert "failed" in caplog.text


def test_training_page_tml_only_end_to_end(tmp_path, monkeypatch):
    from pathlib import Path

    from polynet.config.io import save_options
    import polynet.config.paths as paths
    from polynet.config.schemas import DataConfig, GeneralConfig, RepresentationConfig
    import polynet.experiment.manager as manager
    from polynet.pipeline import compute_descriptors

    monkeypatch.setattr(paths, "polynet_experiments_base_dir", lambda: tmp_path)
    monkeypatch.setattr(manager, "polynet_experiments_base_dir", lambda: tmp_path)
    exp = tmp_path / "gui-test"
    exp.mkdir()
    df = pd.DataFrame(
        {
            "id": [f"m{i}" for i in range(30)],
            "smiles": ["C", "CC", "CCC", "CCO", "CCN"] * 6,
            "target": [float(i % 7) for i in range(30)],
        }
    ).set_index("id")
    csv = exp / "data.csv"
    df.to_csv(csv)
    data = DataConfig(
        data_name="data.csv",
        data_path=str(csv),
        id_col="id",
        smiles_cols=["smiles"],
        target_variable_col="target",
        target_variable_name="target",
        problem_type="regression",
        num_classes=1,
    )
    rep = RepresentationConfig(
        smiles_merge_approach="no_merging", molecular_descriptors={"rdkit": ["MolWt", "TPSA"]}
    )
    save_options(paths.data_options_path(exp), data)
    save_options(paths.general_options_path(exp), GeneralConfig(name="gui-test"))
    save_options(paths.representation_options_path(exp), rep)
    compute_descriptors(df, data, rep, exp)
    page = Path(__file__).parents[1] / "polynet/app/pages/3_Train_Models.py"
    app = AppTest.from_file(str(page), default_timeout=30).run()
    app.selectbox(key="ExperimentName").set_value("gui-test").run()
    assert not app.exception
    app.toggle(key="TrainTML").set_value(True).run()
    app.toggle(key="LinearRegression").set_value(True).run()
    app.button[0].click().run()
    assert not app.exception
    assert paths.ml_results_file_path(exp).exists()
    assert app.status[0].state == "complete"
    assert "split/bootstrap 1/1" in app.code[0].value
    predictions = read_csv(paths.ml_results_file_path(exp))
    # Rerunning requires acknowledgement and invalidates downstream outputs.
    stale_paths = [
        exp / "explanations" / "analyses" / "old.json",
        exp / "unseen_predictions" / "unseen" / "predictions.csv",
        exp / "gnn_hyp_opt" / "old_search.csv",
    ]
    for path in stale_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old")
    app.run()
    assert app.button[0].disabled and all(path.exists() for path in stale_paths)
    confirmation = next(
        checkbox for checkbox in app.checkbox if checkbox.key.startswith("rerun_ack:")
    )
    confirmation.check().run()
    app.button[0].click().run()
    assert not app.exception
    assert all(not path.exists() for path in stale_paths)
    pd.testing.assert_frame_equal(read_csv(paths.ml_results_file_path(exp)), predictions)


@pytest.mark.parametrize("train_tml", [True, False])
def test_preprocessing_gui_defaults(train_tml):
    from polynet.app.components.forms.train_models import default_feature_preprocessing
    from polynet.config.enums import FeatureSelection, TransformDescriptor

    def app_script(train_tml):
        import streamlit as st

        from polynet.app.components.forms.train_models import feature_transformer_widgets

        st.session_state["config"] = feature_transformer_widgets(train_tml, True)

    app = AppTest.from_function(app_script, kwargs={"train_tml": train_tml}).run()
    assert not app.exception
    expected = default_feature_preprocessing(train_tml)
    assert app.session_state["config"] == expected
    assert expected.scaler == TransformDescriptor.MinMaxScaler
    assert list(expected.selectors) == (
        [FeatureSelection.Variance, FeatureSelection.Correlation] if train_tml else []
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_external_results_layout(tmp_path, legacy):
    import json

    root = tmp_path / "unseen_predictions" / "SNR_data_all"
    destination = root / "ml_results" if legacy else root
    destination.mkdir(parents=True)
    pd.DataFrame({"prediction": [1.2, 2.3]}).to_csv(destination / "predictions.csv")
    metrics = {
        "1": {"GCN": {"SNR_data_all": {"rmse": 0.1}}},
        "ensemble": {"GNN": {"SNR_data_all": {"rmse": 0.2}}},
    }
    (destination / "metrics.json").write_text(json.dumps(metrics))

    def app_script(path):
        from pathlib import Path

        from polynet.app.components.plots import display_unseen_predictions

        display_unseen_predictions(Path(path))

    app = AppTest.from_function(app_script, args=(str(tmp_path),)).run()
    assert not app.dataframe
    app.toggle[0].set_value(True).run()
    assert not app.exception
    assert len(app.dataframe) == 3
    assert len(app.dataframe[0].value) == 2
    assert set(app.dataframe[1].value["Iteration"]) == {"1", "ensemble"}


def test_empty_results_panel_explains_missing_files(tmp_path):
    def app_script(path):
        from pathlib import Path

        from polynet.app.components.plots import display_model_results

        display_model_results(Path(path), expanded=True, external=True)

    app = AppTest.from_function(app_script, args=(str(tmp_path),)).run()
    assert not app.exception
    assert "No saved predictions" in app.info[0].value


def test_parity_legend_has_readable_background():
    from matplotlib.colors import to_rgba
    import matplotlib.pyplot as plt

    from polynet.visualization.plots import plot_parity

    with plt.rc_context(
        {
            "axes.facecolor": (0, 0, 0, 0),
            "legend.facecolor": "black",
            "legend.framealpha": 0.2,
            "legend.frameon": False,
            "legend.shadow": True,
            "font.weight": "bold",
        }
    ):
        fig = plot_parity([1, 2, 3], [1.1, 1.9, 3.1])
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend.get_frame_on()
    assert not legend.shadow
    assert legend.get_frame().get_facecolor() == to_rgba("white")
    assert legend.get_texts()[0].get_fontweight() == "normal"
    assert ax.lines[0].get_color() == "#777777"
    assert list(ax.lines[0].get_ydata()) == list(ax.lines[0].get_xdata())
    plt.close(fig)


def test_descriptor_defaults_follow_string_type():
    from rdkit.Chem import Descriptors

    from polynet.app.components.forms.representation import default_molecular_descriptors
    from polynet.config.enums import MolecularDescriptor, PMXAggMethod
    from polynet.config.schemas import RepresentationConfig
    from polynet.featurizer.pmx import (
        BACKBONE_TOPOLOGICAL_FEATURIZER_REGISTRY,
        CHEMICAL_FEATURIZER_REGISTRY,
        SIDECHAIN_TOPOLOGICAL_FEATURIZER_REGISTRY,
    )

    smiles = default_molecular_descriptors("smiles")
    psmiles = default_molecular_descriptors("psmiles")
    assert set(smiles) == {MolecularDescriptor.RDKit, MolecularDescriptor.Morgan}
    assert set(smiles[MolecularDescriptor.RDKit]) == {n for n, _ in Descriptors.descList}
    assert set(psmiles) == {MolecularDescriptor.PolyBERT, MolecularDescriptor.PolyMetriX}
    pmx = psmiles[MolecularDescriptor.PolyMetriX]
    assert set(pmx["polymer"]) == set(CHEMICAL_FEATURIZER_REGISTRY)
    assert set(pmx["side_chain"]) == set(CHEMICAL_FEATURIZER_REGISTRY) | set(
        SIDECHAIN_TOPOLOGICAL_FEATURIZER_REGISTRY
    )
    assert set(pmx["backbone"]) == set(CHEMICAL_FEATURIZER_REGISTRY) | set(
        BACKBONE_TOPOLOGICAL_FEATURIZER_REGISTRY
    )
    assert pmx["agg"] == [PMXAggMethod.Mean]
    for config in (smiles, psmiles):
        RepresentationConfig(smiles_merge_approach="no_merging", molecular_descriptors=config)


def test_switching_to_smiles_clears_polymer_only_selections():
    def app_script():
        import pandas as pd
        import streamlit as st

        from polynet.app.components.forms.representation import molecular_descriptor_representation
        from polynet.config.schemas import DataConfig

        kind = st.selectbox("String type", ["psmiles", "smiles"])
        cfg = DataConfig(
            data_name="data.csv",
            data_path="data.csv",
            id_col="id",
            smiles_cols=["smiles"],
            target_variable_col="target",
            target_variable_name="target",
            problem_type="regression",
            num_classes=1,
            string_representation=kind,
        )
        st.session_state["descriptors"] = molecular_descriptor_representation(
            pd.DataFrame({"smiles": ["*CC*" if kind == "psmiles" else "CC"], "target": [1]}), cfg
        )

    app = AppTest.from_function(app_script, default_timeout=30).run()
    assert not app.exception
    assert set(app.session_state["descriptors"]) == {"polybert", "polymetrix"}
    app.selectbox[0].set_value("smiles").run()
    assert not app.exception
    assert set(app.session_state["descriptors"]) == {"rdkit", "morgan"}
    assert not app.checkbox(key="polyBERTfp").value
    assert app.checkbox(key="polyBERTfp").disabled


def test_training_exports_readable_parity_legend(tmp_path, monkeypatch):
    from PIL import Image
    from matplotlib.colors import to_rgba
    import matplotlib.pyplot as plt
    import pandas as pd

    from polynet.config.column_names import get_iterator_name
    from polynet.config.constants import DataSet, ResultColumn
    from polynet.config.enums import ProblemType, SplitType
    from polynet.training.evaluate import plot_results
    from polynet.visualization import utils

    save_original = utils.save_plot
    captured = []

    def save(fig, path):
        legend = fig.axes[0].get_legend()
        captured.append((legend.get_frame_on(), legend.get_frame().get_facecolor(), legend.shadow))
        save_original(fig, path)

    monkeypatch.setattr(utils, "save_plot", save)
    predictions = pd.DataFrame(
        {
            get_iterator_name(SplitType.TrainValTest): ["1"] * 3,
            ResultColumn.SET: [DataSet.Test] * 3,
            "True target": [1.0, 2.0, 3.0],
            "GCN Predicted target": [1.1, 1.9, 3.2],
        }
    )
    with plt.rc_context(
        {
            "legend.facecolor": "black",
            "legend.framealpha": 0.2,
            "legend.frameon": False,
            "legend.shadow": True,
        }
    ):
        plot_results(
            predictions,
            SplitType.TrainValTest,
            "target",
            ["GCN_1"],
            ProblemType.Regression,
            tmp_path,
        )
    assert captured == [(True, to_rgba("white"), False)]
    with Image.open(tmp_path / "GCN_1_parity_plot.png") as image:
        assert image.width > 100 and image.height > 100
        assert len(image.getcolors(image.width * image.height)) > 10
