"""
tests/test_hpo_cv.py
====================
Cross-validation settings shared by the TML and GNN hyperparameter searches:

- ``hpo_n_folds`` is one schema field (``HyperparamOptimConfig``) with ``k >= 2``;
- ``polynet.training.cv`` builds the shuffled (stratified) splitter for both;
- ``polynet.pipeline.validate_hpo_folds`` checks ``k`` against the data of every
  split before any training, for whichever pipeline will run cross-validated HPO.
"""

import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from polynet.config.enums import HpoSplitStrategy, Network, ProblemType, TraditionalMLModel
from polynet.config.schemas import DataConfig, TrainGNNConfig, TrainTMLConfig
from polynet.pipeline import validate_hpo_folds
from polynet.training.cv import check_n_folds, check_n_folds_for_splits, make_kfold

_TML_HPO = {TraditionalMLModel.RandomForest: {}}
_GNN_HPO = {Network.GCN: {}}

# ---------------------------------------------------------------------------
# Schema: one shared field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_cfg",
    [
        lambda **kw: TrainTMLConfig(train_tml=True, selected_models=_TML_HPO, **kw),
        lambda **kw: TrainGNNConfig(gnn_convolutional_layers=_GNN_HPO, **kw),
    ],
    ids=["tml", "gnn"],
)
def test_schema_shares_hpo_n_folds(make_cfg):
    assert make_cfg().hpo_n_folds == 5
    assert make_cfg(hpo_n_folds=7).hpo_n_folds == 7
    for k in (0, 1, -3):
        with pytest.raises(ValidationError, match="hpo_n_folds"):
            make_cfg(hpo_n_folds=k)


# ---------------------------------------------------------------------------
# Shared splitter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "problem_type, splitter",
    [(ProblemType.Regression, KFold), (ProblemType.Classification, StratifiedKFold)],
)
def test_make_kfold_is_shuffled_and_seeded(problem_type, splitter):
    cv = make_kfold(problem_type=problem_type, n_folds=4, random_seed=7)
    assert isinstance(cv, splitter)
    assert (cv.n_splits, cv.shuffle, cv.random_state) == (4, True, 7)


def test_folds_are_not_contiguous_when_target_is_sorted():
    """Unshuffled KFold on a target-sorted dataset gives each fold one slice of
    the target range; shuffled folds must each span most of the range."""
    y = np.arange(100, dtype=float)
    cv = make_kfold(problem_type=ProblemType.Regression, n_folds=5, random_seed=0)
    for _, test_idx in cv.split(y.reshape(-1, 1), y):
        assert y[test_idx].max() - y[test_idx].min() > 50


def test_gnn_hpo_uses_the_shared_splitter():
    from polynet.training import hyperopt

    class _G:
        def __init__(self, y):
            self.y = __import__("torch").tensor(float(y))

    dataset = [_G(v) for v in range(20)]
    splits = hyperopt._build_splits(
        dataset=dataset,
        problem_type=ProblemType.Regression,
        strategy=HpoSplitStrategy.CrossValidation,
        random_seed=0,
        n_folds=4,
    )
    expected = make_kfold(ProblemType.Regression, n_folds=4, random_seed=0).split(np.zeros(20))
    assert [va for _, va in splits] == [va.tolist() for _, va in expected]


# ---------------------------------------------------------------------------
# Data-aware checks
# ---------------------------------------------------------------------------


def test_check_rejects_more_folds_than_samples():
    with pytest.raises(ValueError, match=r"exceed the number of samples \(10\).*between 2 and 10"):
        check_n_folds(n_folds=11, y=np.arange(10.0), problem_type=ProblemType.Regression)


def test_check_rejects_more_folds_than_smallest_class():
    y = np.array([0] * 20 + [1] * 3)
    with pytest.raises(ValueError, match=r"class 1 has 3 samples.*between 2 and 3"):
        check_n_folds(n_folds=4, y=y, problem_type=ProblemType.Classification)


@pytest.mark.parametrize(
    "k, y, problem_type",
    [
        (10, np.arange(10.0), ProblemType.Regression),
        (3, np.array([0] * 20 + [1] * 3), ProblemType.Classification),
    ],
)
def test_check_accepts_boundary_values(k, y, problem_type):
    check_n_folds(n_folds=k, y=y, problem_type=problem_type)


def test_split_check_names_setting_and_split():
    y = pd.Series([0] * 10 + [1] * 10, index=[f"id{i}" for i in range(20)])
    ok_split = [f"id{i}" for i in range(20)]
    bad_split = [f"id{i}" for i in range(12)]  # only 2 samples of class 1
    with pytest.raises(ValueError, match=r"gnn_training.hpo_n_folds=3 is invalid for split 2"):
        check_n_folds_for_splits(
            n_folds=3,
            y=y,
            train_val_test_idxs=([ok_split, bad_split], [[], []], [[], []]),
            problem_type=ProblemType.Classification,
            setting_name="gnn_training.hpo_n_folds",
        )


# ---------------------------------------------------------------------------
# Pipeline stage: one check for both pipelines
# ---------------------------------------------------------------------------

_DATA = pd.DataFrame({"target": [0] * 16 + [1] * 4}, index=range(20))
_SPLITS = ([list(range(0, 16, 2)) + [16, 17]], [[1, 3]], [[5, 18, 19]])  # HPO: 2 of class 1


def _data_cfg():
    return DataConfig(
        data_name="d.csv",
        data_path="d.csv",
        smiles_cols=["smiles"],
        target_variable_col="target",
        problem_type=ProblemType.Classification,
        num_classes=2,
    )


def test_stage_rejects_bad_tml_k():
    tml_cfg = TrainTMLConfig(train_tml=True, selected_models=_TML_HPO, hpo_n_folds=3)
    with pytest.raises(ValueError, match="tml_models.hpo_n_folds=3 is invalid for split 1"):
        validate_hpo_folds(_DATA, _data_cfg(), _SPLITS, tml_cfg=tml_cfg)


def test_stage_rejects_bad_gnn_k_under_cross_validation():
    gnn_cfg = TrainGNNConfig(gnn_convolutional_layers=_GNN_HPO, hpo_n_folds=3)
    with pytest.raises(ValueError, match="gnn_training.hpo_n_folds=3 is invalid for split 1"):
        validate_hpo_folds(_DATA, _data_cfg(), _SPLITS, gnn_cfg=gnn_cfg)


def test_stage_skips_pipelines_without_cross_validated_hpo():
    # TML with explicit hyperparameters → no HPO.
    tml_cfg = TrainTMLConfig(
        train_tml=True,
        selected_models={TraditionalMLModel.RandomForest: {"n_estimators": 10}},
        hpo_n_folds=3,
    )
    # GNN HPO with a holdout strategy → no folds.
    gnn_cfg = TrainGNNConfig(
        gnn_convolutional_layers=_GNN_HPO, hpo_split_strategy=HpoSplitStrategy.Holdout
    )
    validate_hpo_folds(_DATA, _data_cfg(), _SPLITS, tml_cfg=tml_cfg, gnn_cfg=gnn_cfg)


def test_stage_accepts_valid_k():
    tml_cfg = TrainTMLConfig(train_tml=True, selected_models=_TML_HPO, hpo_n_folds=2)
    gnn_cfg = TrainGNNConfig(gnn_convolutional_layers=_GNN_HPO, hpo_n_folds=2)
    validate_hpo_folds(_DATA, _data_cfg(), _SPLITS, tml_cfg=tml_cfg, gnn_cfg=gnn_cfg)
