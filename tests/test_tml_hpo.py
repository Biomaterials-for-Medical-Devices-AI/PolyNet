"""
tests/test_tml_hpo.py
=====================
TML hyperparameter search: a randomised search (``RandomizedSearchCV``)
scored with the shared shuffled K-fold splitter and the user's ``k``.
"""

from unittest import mock

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from polynet.config.enums import ProblemType, TraditionalMLModel
import polynet.training.tml as tml


def test_old_grid_search_name_is_gone():
    assert hasattr(tml, "_run_random_search")
    assert not hasattr(tml, "_run_grid_search")


def _run(n_folds: int = 5, random_seed: int = 3):
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
    train_df["y"] = np.sort(rng.normal(size=40))  # sorted target

    fake_search = mock.MagicMock(best_params_={}, best_estimator_="best")
    with mock.patch.object(tml, "RandomizedSearchCV", return_value=fake_search) as rscv:
        best = tml._run_random_search(
            model=mock.MagicMock(),
            model_id=TraditionalMLModel.RandomForest,
            train_df=train_df,
            problem_type=ProblemType.Regression,
            random_seed=random_seed,
            n_folds=n_folds,
        )
    fake_search.fit.assert_called_once()
    return best, rscv.call_args.kwargs


def test_random_search_uses_shuffled_cv_and_returns_best_estimator():
    best, kwargs = _run(random_seed=3)
    assert isinstance(kwargs["cv"], KFold)
    assert kwargs["cv"].shuffle is True and kwargs["cv"].random_state == 3
    assert kwargs["n_iter"] == 30
    assert kwargs["random_state"] == 3
    assert best == "best"


def test_random_search_uses_user_fold_count():
    _, kwargs = _run(n_folds=8)
    assert kwargs["cv"].n_splits == 8
