"""
tests/test_hpo_search_grid.py
=============================
User-defined HPO search grids (``hpo_search_grid``) and sample counts
(``hpo_num_samples``) for the GNN and TML pipelines: validation at config
load, merging on top of the default grids, n_iter capping, pass-through to
the search libraries, provenance, and the GNN HPO cache.
"""

import json
import logging
import types
from unittest import mock
import warnings

import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest

torch = pytest.importorskip("torch")

from polynet.config.enums import (  # noqa: E402
    ArchitectureParam,
    HpoSplitStrategy,
    Network,
    ProblemType,
    TraditionalMLModel,
    TrainingParam,
)
from polynet.config.schemas import TrainGNNConfig, TrainTMLConfig  # noqa: E402
from polynet.config.search_grid import get_gnn_search_grid, get_tml_search_grid  # noqa: E402
import polynet.training.hyperopt as hyperopt  # noqa: E402
import polynet.training.tml as tml  # noqa: E402

# ---------------------------------------------------------------------------
# Validation at config load
# ---------------------------------------------------------------------------


def _gnn(grid, layers=None):
    return TrainGNNConfig(gnn_convolutional_layers=layers or {"GCN": {}}, hpo_search_grid=grid)


def _tml(grid, models=None):
    return TrainTMLConfig(
        train_tml=True, selected_models=models or {"random_forest": {}}, hpo_search_grid=grid
    )


def test_valid_overrides_and_default_sample_counts():
    gnn = _gnn({"shared": {"dropout": [0.2]}, "GCN": {"improved": [True]}})
    tml_cfg = _tml({"random_forest": {"n_estimators": [50, 100]}})
    assert gnn.hpo_num_samples == 150 and tml_cfg.hpo_num_samples == 30


@pytest.mark.parametrize(
    "make, match",
    [
        (lambda: _gnn({"GCN": {"num_heads": [2]}}), r"GCN\.num_heads is not a searchable"),
        (lambda: _gnn({"GCN": {"dropout": []}}), "non-empty list"),
        (lambda: _gnn({"GCN": {"dropout": 0.1}}), "non-empty list"),
        (lambda: _gnn({"MPNN": {"dropout": [0.1]}}), "not a selected architecture"),
        (lambda: _gnn({"shared": {"seed": [1]}}), "set by PolyNet"),
        (lambda: _tml({"random_forest": {"n_trees": [3]}}), r"random_forest\.n_trees is not"),
        (lambda: _tml({"xgboost": {"max_depth": [3]}}), "not a selected model"),
        (lambda: _tml({"random_forest": {"random_state": [1]}}), "set by PolyNet"),
        (lambda: TrainGNNConfig(gnn_convolutional_layers={"GCN": {}}, hpo_num_samples=0), "hpo_num_samples"),
    ],
)
def test_invalid_grids_are_rejected(make, match):
    with pytest.raises(ValidationError, match=match):
        make()


def test_grid_for_model_with_explicit_hyperparameters_warns():
    with pytest.warns(UserWarning, match="has no effect"):
        _tml({"random_forest": {"max_depth": [3]}}, models={"random_forest": {"n_estimators": 5}})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _tml({"random_forest": {"max_depth": [3]}})  # HPO model → no warning


# ---------------------------------------------------------------------------
# Merge on top of the defaults
# ---------------------------------------------------------------------------


def test_gnn_override_replaces_only_the_given_parameters():
    default = get_gnn_search_grid(Network.GCN, random_seed=7)
    custom = {"shared": {"dropout": [0.2], "embedding_dim": [16]}, "GCN": {"embedding_dim": [8]}}
    merged = get_gnn_search_grid(Network.GCN, random_seed=7, custom_grid=custom)

    assert merged[ArchitectureParam.Dropout] == [0.2]  # from shared
    assert merged[ArchitectureParam.EmbeddingDim] == [8]  # architecture wins over shared
    assert merged[TrainingParam.Seed] == [7]  # seed still injected
    untouched = set(default) - {ArchitectureParam.Dropout, ArchitectureParam.EmbeddingDim}
    assert all(merged[k] == default[k] for k in untouched)
    assert list(merged) == list(default)  # key order kept (Ray sampling order)


def test_no_custom_grid_gives_the_default_grid():
    assert get_gnn_search_grid(Network.GAT, 3, custom_grid={}) == get_gnn_search_grid(Network.GAT, 3)
    assert get_tml_search_grid(
        TraditionalMLModel.XGBoost, ProblemType.Regression, 3, custom_grid=None
    ) == get_tml_search_grid(TraditionalMLModel.XGBoost, ProblemType.Regression, 3)


def test_tml_override_keeps_defaults_and_injected_seed():
    grid = get_tml_search_grid(
        TraditionalMLModel.RandomForest, ProblemType.Regression, 3, {"n_estimators": [50]}
    )
    assert grid["n_estimators"] == [50]
    assert grid["max_depth"] == [None, 3, 6] and grid["random_state"] == [3]


# ---------------------------------------------------------------------------
# TML: n_iter capping and pass-through
# ---------------------------------------------------------------------------


def test_n_iter_is_capped_at_the_number_of_combinations(caplog):
    with caplog.at_level(logging.WARNING):
        grid, n_iter = tml.tml_search_space(
            TraditionalMLModel.LinearRegression, ProblemType.Regression, 0, n_iter=30
        )
    assert grid == {"fit_intercept": [True, False]} and n_iter == 2
    assert "sampling all 2" in caplog.text
    _, n_iter = tml.tml_search_space(
        TraditionalMLModel.RandomForest, ProblemType.Regression, 0, n_iter=30
    )
    assert n_iter == 30  # 81 combinations → not capped


def test_tml_search_receives_user_grid_and_sample_count():
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "y"])
    fake = mock.MagicMock(best_params_={}, best_estimator_=types.SimpleNamespace(), best_score_=0.1)
    with mock.patch.object(tml, "RandomizedSearchCV", return_value=fake) as rscv:
        best = tml._run_random_search(
            model=mock.MagicMock(),
            model_id=TraditionalMLModel.RandomForest,
            train_df=train_df,
            problem_type=ProblemType.Regression,
            random_seed=4,
            n_iter=7,
            custom_grid={"n_estimators": [10, 20]},
        )
    kwargs = rscv.call_args.kwargs
    assert kwargs["n_iter"] == 7
    assert kwargs["param_distributions"]["n_estimators"] == [10, 20]
    assert best.polynet_hpo_["search_grid"]["n_estimators"] == [10, 20]
    assert best.polynet_hpo_["n_iter"] == 7


# ---------------------------------------------------------------------------
# GNN: pass-through, provenance and cache
# ---------------------------------------------------------------------------


def _fake_tune_run(*args, **kwargs):
    config = {k: v.categories[0] for k, v in kwargs["config"].items()}
    return types.SimpleNamespace(
        get_best_trial=lambda *a: types.SimpleNamespace(config=config),
        results_df=pd.DataFrame(
            [{"val_loss": 0.5, **{f"config/{k}": v for k, v in config.items()}}]
        ),
    )


def _run_gnn_hpo(tmp_path, **overrides):
    dataset = [types.SimpleNamespace(y=torch.tensor(float(i))) for i in range(20)]
    kwargs = dict(
        exp_path=tmp_path,
        gnn_arch=Network.GCN,
        dataset=dataset,
        num_classes=1,
        num_samples=12,
        iteration=1,
        problem_type=ProblemType.Regression,
        random_seed=0,
        hpo_split_strategy=HpoSplitStrategy.Holdout,
        custom_grid={"GCN": {"embedding_dim": [16]}},
    )
    kwargs.update(overrides)
    with mock.patch.object(hyperopt.ray, "init"), mock.patch.object(
        hyperopt.ray, "shutdown"
    ), mock.patch.object(hyperopt.tune, "run", side_effect=_fake_tune_run) as run:
        best = hyperopt.gnn_hyp_opt(**kwargs)
    return best, run


def test_gnn_hpo_receives_sample_count_and_merged_grid(tmp_path):
    best, run = _run_gnn_hpo(tmp_path)
    kwargs = run.call_args.kwargs
    assert kwargs["num_samples"] == 12
    assert kwargs["config"][ArchitectureParam.EmbeddingDim].categories == [16]
    assert best[ArchitectureParam.EmbeddingDim] == 16

    (space_file,) = tmp_path.glob("gnn_hyp_opt/iteration_1/GCN_*/search_space.json")
    space = json.loads(space_file.read_text())
    assert space["num_samples"] == 12 and space["search_grid"]["embedding_dim"] == [16]


def test_gnn_cache_is_reused_only_for_the_same_search(tmp_path):
    _, first = _run_gnn_hpo(tmp_path)
    _, same = _run_gnn_hpo(tmp_path)
    assert first.call_count == 1 and same.call_count == 0  # identical search → cached

    _, new_grid = _run_gnn_hpo(tmp_path, custom_grid={"GCN": {"embedding_dim": [32]}})
    _, new_samples = _run_gnn_hpo(tmp_path, num_samples=13)
    assert new_grid.call_count == 1 and new_samples.call_count == 1  # changed → re-run
    assert len(list(tmp_path.glob("gnn_hyp_opt/iteration_1/GCN_*"))) == 3


# ---------------------------------------------------------------------------
# Experiment-level summary of the search spaces (hpo_search_spaces.json)
# ---------------------------------------------------------------------------


def test_effective_search_spaces_lists_only_tuned_models_with_merged_grids():
    from polynet.config.search_grid import effective_search_spaces

    gnn = TrainGNNConfig(
        gnn_convolutional_layers={"GCN": {}, "GAT": {"learning_rate": 0.01, "batch_size": 8}},
        hpo_num_samples=20,
        hpo_search_grid={"shared": {"dropout": [0.3]}},
    )
    tml_cfg = TrainTMLConfig(
        train_tml=True,
        selected_models={"linear_regression": {}, "random_forest": {"n_estimators": 5}},
    )
    spaces = effective_search_spaces(ProblemType.Regression, gnn_cfg=gnn, tml_cfg=tml_cfg)

    assert list(spaces["gnn"]["architectures"]) == ["GCN"]  # GAT has explicit params
    gcn = spaces["gnn"]["architectures"]["GCN"]
    assert gcn[ArchitectureParam.Dropout] == [0.3] and TrainingParam.Seed not in gcn
    assert spaces["gnn"]["hpo_num_samples"] == 20

    assert list(spaces["tml"]["models"]) == ["linear_regression"]
    lr = spaces["tml"]["models"]["linear_regression"]
    assert lr == {"search_grid": {"fit_intercept": [True, False]}, "n_iter": 2}  # capped

    assert effective_search_spaces(ProblemType.Regression) == {}


def test_gnn_hpo_trials_use_the_training_epochs(tmp_path):
    with mock.patch.object(hyperopt, "ASHAScheduler") as asha, mock.patch.object(
        hyperopt.tune, "with_parameters"
    ) as trial:
        _run_gnn_hpo(tmp_path, epochs=40)
    # ASHA (holdout) stops at training.epochs; trials receive the same epochs.
    assert asha.call_args.kwargs["max_t"] == 40
    assert asha.call_args.kwargs["grace_period"] == hyperopt.asha_grace_period(40) == 8
    assert trial.call_args.kwargs["epochs"] == 40

    _, again = _run_gnn_hpo(tmp_path, epochs=60)
    assert again.call_count == 1  # a different number of epochs is a new search


def test_trials_train_for_the_requested_epochs(monkeypatch):
    calls = []
    monkeypatch.setattr(hyperopt, "train_network", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(hyperopt, "eval_network", lambda *a, **k: 1.0)
    monkeypatch.setattr(hyperopt, "create_network", lambda **k: mock.MagicMock())
    monkeypatch.setattr(hyperopt, "fit_polymer_descriptor_scaler", lambda *a: None)
    monkeypatch.setattr(hyperopt, "n_polymer_descriptors_of", lambda g: 0)
    monkeypatch.setattr(hyperopt, "build_optimisation", lambda **k: (None, None, None))
    monkeypatch.setattr(hyperopt, "step_scheduler", lambda *a: None)
    reports = []
    monkeypatch.setattr(hyperopt.session, "report", reports.append)
    dataset = [types.SimpleNamespace(num_node_features=3, num_edge_features=1) for _ in range(10)]
    config = {TrainingParam.LearningRate: 0.01, TrainingParam.BatchSize: 4}

    for strategy, n_splits in ((HpoSplitStrategy.CrossValidation, 2), (HpoSplitStrategy.Holdout, 1)):
        calls.clear()
        hyperopt._gnn_target_function(
            config=dict(config),
            dataset=dataset,
            num_classes=1,
            splits=[(list(range(7)), [7, 8, 9])] * n_splits,
            strategy=strategy,
            network=Network.GCN,
            problem_type=ProblemType.Regression,
            epochs=7,
        )
        assert len(calls) == 7 * n_splits
