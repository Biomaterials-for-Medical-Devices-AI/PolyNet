"""
tests/test_tml_validation_option.py
===================================
``tml_models.include_validation_in_training``: TML models are trained on the
training + validation samples (default) or on the training samples only — the
data GNNs train on — with the validation samples scored as a held-out set.
"""

import numpy as np
import pandas as pd
import pytest

from polynet.config.constants import DataSet, ResultColumn
from polynet.config.enums import (
    ProblemType,
    SplitType,
    TargetTransformDescriptor,
    TraditionalMLModel,
    TransformDescriptor,
)
from polynet.config.schemas import TrainTMLConfig
from polynet.inference.tml import get_predictions_df_tml
from polynet.training.metrics import get_metrics
from polynet.training.tml import train_tml_ensemble
from polynet.utils.validation import check_n_folds_for_splits

_RNG = np.random.default_rng(0)
_DF = pd.DataFrame(
    _RNG.normal(size=(60, 3)), columns=["a", "b", "c"], index=[f"s{i}" for i in range(60)]
)
_DF["y"] = _DF["a"] * 2 + _RNG.normal(scale=0.1, size=60)
_TRAIN, _VAL, _TEST = list(_DF.index[:40]), list(_DF.index[40:50]), list(_DF.index[50:])


def _train(include: bool):
    return train_tml_ensemble(
        tml_models={TraditionalMLModel.LinearRegression: {}},
        problem_type=ProblemType.Regression,
        transform_type=TransformDescriptor.StandardScaler,
        feature_selection={},
        dataframes={"desc": _DF},
        random_seed=0,
        train_val_test_idxs=([pd.Index(_TRAIN)], [pd.Index(_VAL)], [pd.Index(_TEST)]),
        target_transform=TargetTransformDescriptor.StandardScaler,
        hpo_num_samples=2,
        include_validation_in_training=include,
    )


def test_default_is_to_include_validation():
    cfg = TrainTMLConfig(train_tml=True, selected_models={TraditionalMLModel.RandomForest: {}})
    assert cfg.include_validation_in_training is True


@pytest.mark.parametrize("include, fitted_on", [(True, _TRAIN + _VAL), (False, _TRAIN)])
def test_scalers_and_models_are_fitted_on_the_chosen_samples(include, fitted_on):
    models, data, scalers, target_scalers = _train(include)
    train_df, val_df, _ = data["desc_1"]
    assert list(train_df.index) == fitted_on
    assert (val_df is None) == include
    # Feature scaler: training samples have mean 0 only for the samples it was fitted on.
    assert np.allclose(train_df[["a", "b", "c"]].mean(), 0, atol=1e-9)
    # Target scaler fitted on the same samples.
    assert target_scalers["desc_1"].transform(
        _DF.loc[fitted_on, "y"].values
    ).mean() == pytest.approx(0, abs=1e-9)
    # The model was fitted on exactly these samples: refitting on them gives
    # the same coefficients.
    from sklearn.linear_model import LinearRegression

    refit = LinearRegression().fit(
        train_df[["a", "b", "c"]], target_scalers["desc_1"].transform(train_df["y"].values)
    )
    assert np.allclose(models["linear_regression-desc_1"].coef_, refit.coef_)


def test_excluded_validation_is_predicted_and_scored_as_validation():
    models, data, _, target_scalers = _train(include=False)
    preds = get_predictions_df_tml(
        models, data, SplitType.TrainValTest, "y", ProblemType.Regression, "y", target_scalers
    )
    sets = preds.groupby(ResultColumn.SET)[ResultColumn.INDEX].apply(set)
    assert sets[DataSet.Validation] == set(_VAL) and sets[DataSet.Training] == set(_TRAIN)

    metrics = get_metrics(preds, SplitType.TrainValTest, "y", list(models), ProblemType.Regression)
    assert set(metrics["1"]["linear_regression-desc"]) == {
        DataSet.Training,
        DataSet.Validation,
        DataSet.Test,
    }


def test_included_validation_is_reported_as_training():
    models, data, _, target_scalers = _train(include=True)
    preds = get_predictions_df_tml(
        models, data, SplitType.TrainValTest, "y", ProblemType.Regression, "y", target_scalers
    )
    assert set(preds[ResultColumn.SET]) == {DataSet.Training, DataSet.Test}
    assert set(preds.loc[preds[ResultColumn.SET] == DataSet.Training, ResultColumn.INDEX]) == set(
        _TRAIN + _VAL
    )


def test_hpo_fold_check_counts_only_training_samples_when_validation_is_excluded():
    y = pd.Series([0] * 3 + [1] * 37 + [0] * 10, index=[f"s{i}" for i in range(50)])
    splits = ([y.index[:40]], [y.index[40:]], [[]])
    check_n_folds_for_splits(5, y, splits, ProblemType.Classification)  # 13 of class 0 with val
    with pytest.raises(ValueError, match="hpo_n_folds=5"):  # only 3 of class 0 without
        check_n_folds_for_splits(5, y, splits, ProblemType.Classification, include_validation=False)
