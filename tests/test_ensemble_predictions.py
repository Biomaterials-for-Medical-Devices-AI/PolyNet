"""
tests/test_ensemble_predictions.py
==================================
Ensembles over the models trained on the repeated splits: mean ± standard
deviation for regression, majority vote + vote fraction for classification.
"""

import numpy as np
import pandas as pd
import pytest

from polynet.config.enums import ProblemType
from polynet.inference.ensemble import ensemble_predictions

T = "Tg"


def test_regression_mean_and_population_std():
    preds = pd.DataFrame(
        {"GCN 1 Predicted Tg": [1.0, 10.0], "GCN 2 Predicted Tg": [3.0, 10.0], "x": [0, 0]}
    )
    out, cols = ensemble_predictions(
        preds, {"GCN": ["GCN 1 Predicted Tg", "GCN 2 Predicted Tg"]}, ProblemType.Regression, T
    )
    np.testing.assert_allclose(out["GCN Ensemble Predicted Tg"], [2.0, 10.0])
    np.testing.assert_allclose(out["GCN Ensemble Std Tg"], [1.0, 0.0])  # ddof=0
    assert cols == {"GCN Ensemble Predicted Tg": "GCN Ensemble"}
    assert list(out.index) == list(preds.index)


def test_classification_majority_vote_and_vote_fraction():
    members = ["rf-desc 1 Predicted Tg", "rf-desc 2 Predicted Tg", "rf-desc 3 Predicted Tg"]
    preds = pd.DataFrame(dict(zip(members, [[1, 0, 1], [1, 0, 0], [0, 0, 1]])))
    out, cols = ensemble_predictions(preds, {"rf-desc": members}, ProblemType.Classification, T)
    assert out["rf-desc Ensemble Predicted Tg"].tolist() == [1, 0, 1]
    np.testing.assert_allclose(out["rf-desc Ensemble Vote Fraction Tg"], [2 / 3, 1.0, 2 / 3])
    assert "rf-desc Ensemble Std Tg" not in out
    assert cols == {"rf-desc Ensemble Predicted Tg": "rf-desc Ensemble"}


def test_classification_tie_goes_to_smallest_label():
    preds = pd.DataFrame({"a": [0, 2], "b": [1, 1]})
    out, _ = ensemble_predictions(preds, {"M": ["a", "b"]}, ProblemType.Classification, T)
    assert out["M Ensemble Predicted Tg"].tolist() == [0, 1]
    np.testing.assert_allclose(out["M Ensemble Vote Fraction Tg"], [0.5, 0.5])


@pytest.mark.parametrize("problem_type", list(ProblemType))
def test_single_member_groups_are_skipped(problem_type):
    preds = pd.DataFrame({"a": [1.0, 0.0]})
    out, cols = ensemble_predictions(preds, {"M": ["a"]}, problem_type, T)
    assert out.empty and cols == {}
