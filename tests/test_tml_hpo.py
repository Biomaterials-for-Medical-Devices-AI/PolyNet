"""
tests/test_tml_hpo.py
=====================
TML hyperparameter search: a randomised search (``RandomizedSearchCV``)
scored with shuffled K-fold CV, so datasets sorted by target do not yield
biased folds.
"""

from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from polynet.config.enums import ProblemType, TraditionalMLModel
import polynet.training.tml as tml


def test_old_grid_search_name_is_gone():
    assert hasattr(tml, "_run_random_search")
    assert not hasattr(tml, "_run_grid_search")


@pytest.mark.parametrize(
    "problem_type, splitter",
    [(ProblemType.Regression, KFold), (ProblemType.Classification, StratifiedKFold)],
)
def test_hpo_cv_is_shuffled_and_seeded(problem_type, splitter):
    cv = tml._make_hpo_cv(problem_type=problem_type, random_seed=7)
    assert isinstance(cv, splitter)
    assert cv.n_splits == 5
    assert cv.shuffle is True
    assert cv.random_state == 7


def test_regression_folds_are_not_contiguous_when_target_is_sorted():
    """Unshuffled KFold on a target-sorted dataset gives each fold one slice of
    the target range; shuffled folds must each span most of the range."""
    y = np.arange(100, dtype=float)  # dataset sorted by target
    X = y.reshape(-1, 1)
    cv = tml._make_hpo_cv(problem_type=ProblemType.Regression, random_seed=0)
    for _, test_idx in cv.split(X, y):
        assert y[test_idx].max() - y[test_idx].min() > 50


def test_random_search_uses_shuffled_cv_and_returns_best_estimator():
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
    train_df["y"] = np.sort(rng.normal(size=40))  # sorted target

    fake_search = mock.MagicMock()
    fake_search.best_params_ = {}
    fake_search.best_estimator_ = "best"
    with mock.patch.object(tml, "RandomizedSearchCV", return_value=fake_search) as rscv:
        best = tml._run_random_search(
            model=mock.MagicMock(),
            model_id=TraditionalMLModel.RandomForest,
            train_df=train_df,
            problem_type=ProblemType.Regression,
            random_seed=3,
        )

    kwargs = rscv.call_args.kwargs
    assert isinstance(kwargs["cv"], KFold)
    assert kwargs["cv"].shuffle is True and kwargs["cv"].random_state == 3
    assert kwargs["n_iter"] == 30
    assert kwargs["random_state"] == 3
    fake_search.fit.assert_called_once()
    assert best == "best"


# ---------------------------------------------------------------------------
# User-controlled number of folds
# ---------------------------------------------------------------------------

from pydantic import ValidationError  # noqa: E402

from polynet.config.schemas.training import TrainTMLConfig  # noqa: E402

_MODELS = {TraditionalMLModel.RandomForest: {}}


def test_schema_default_is_five_folds():
    assert TrainTMLConfig(train_tml=True, selected_models=_MODELS).hpo_n_folds == 5


@pytest.mark.parametrize("k", [0, 1, -3])
def test_schema_rejects_fewer_than_two_folds(k):
    with pytest.raises(ValidationError, match="hpo_n_folds"):
        TrainTMLConfig(train_tml=True, selected_models=_MODELS, hpo_n_folds=k)


def test_validate_rejects_more_folds_than_samples():
    with pytest.raises(ValueError, match="exceeds the number of training samples"):
        tml.validate_hpo_n_folds(n_folds=11, y=np.arange(10.0), problem_type=ProblemType.Regression)


def test_validate_rejects_more_folds_than_smallest_class():
    y = np.array([0] * 20 + [1] * 3)
    with pytest.raises(ValueError, match=r"smallest class .* has 3 samples.*between 2 and 3"):
        tml.validate_hpo_n_folds(n_folds=4, y=y, problem_type=ProblemType.Classification)


@pytest.mark.parametrize(
    "k, y, problem_type",
    [
        (10, np.arange(10.0), ProblemType.Regression),
        (3, np.array([0] * 20 + [1] * 3), ProblemType.Classification),
    ],
)
def test_validate_accepts_boundary_values(k, y, problem_type):
    tml.validate_hpo_n_folds(n_folds=k, y=y, problem_type=problem_type)


def test_random_search_uses_user_fold_count():
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame(rng.normal(size=(40, 2)), columns=["a", "b"])
    train_df["y"] = rng.normal(size=40)

    fake_search = mock.MagicMock(best_params_={}, best_estimator_="best")
    with mock.patch.object(tml, "RandomizedSearchCV", return_value=fake_search) as rscv:
        tml._run_random_search(
            model=mock.MagicMock(),
            model_id=TraditionalMLModel.RandomForest,
            train_df=train_df,
            problem_type=ProblemType.Regression,
            random_seed=0,
            n_folds=8,
        )
    assert rscv.call_args.kwargs["cv"].n_splits == 8


def test_ensemble_fails_fast_before_fitting_on_invalid_k():
    """An impossible k stops the run before any model is fitted."""
    y = [0] * 16 + [1] * 4
    df = pd.DataFrame({"f1": np.arange(20.0), "f2": np.arange(20.0) ** 2, "target": y})
    train_ids, val_ids, test_ids = [list(range(0, 16, 2)) + [16, 17]], [[1, 3]], [[5, 18, 19]]

    with mock.patch.object(tml, "FeatureTransformer") as ft:
        with pytest.raises(ValueError, match="smallest class"):
            tml.train_tml_ensemble(
                tml_models={TraditionalMLModel.RandomForest: {}},
                problem_type=ProblemType.Classification,
                transform_type="no_transformation",
                feature_selection={},
                dataframes={"desc": df},
                random_seed=0,
                train_val_test_idxs=(train_ids, val_ids, test_ids),
                hpo_n_folds=3,
            )
    ft.assert_not_called()


def test_split_validation_names_the_failing_split():
    y = pd.Series([0] * 10 + [1] * 10, index=[f"id{i}" for i in range(20)])
    ok_split = [f"id{i}" for i in range(20)]
    bad_split = [f"id{i}" for i in range(12)]  # only 2 samples of class 1
    with pytest.raises(ValueError, match="invalid for split 2: .*class 1 has 2 samples"):
        tml.validate_hpo_n_folds_for_splits(
            n_folds=3,
            y=y,
            train_val_test_idxs=([ok_split, bad_split], [[], []], [[], []]),
            problem_type=ProblemType.Classification,
        )
