"""
tests/test_shap_class_selection.py
==================================
Regression tests for ``_compute_shap_values_for_row`` class handling.

The bug these guard against: for binary classification the 2-D SHAP output
(LinearExplainer for LogisticRegression, binary TreeExplainer for XGBoost)
silently ignored ``target_class``, so explaining class 0 and class 1 produced
identical values. The correct behaviour — consistent with the 3-D and list
output paths — is that class 0 is the exact mirror (negation) of class 1.
"""

import numpy as np
import pytest

from polynet.config.enums import ProblemType
from polynet.explainability.shap_explain import _compute_shap_values_for_row


class _FakeExplainer:
    """Returns a pre-set SHAP output regardless of input, mimicking shap's API."""

    def __init__(self, output):
        self._output = output

    def shap_values(self, x_2d):
        return self._output


_X = np.array([1.0, 2.0, 3.0])  # one sample, 3 features


# ---------------------------------------------------------------------------
# 2-D output (LinearExplainer / binary TreeExplainer): the fixed branch
# ---------------------------------------------------------------------------


def test_2d_binary_class0_is_negation_of_class1():
    pos = np.array([[0.5, -0.2, 0.1]])  # (1, n_features), positive-class values
    expl = _FakeExplainer(pos)

    c1 = _compute_shap_values_for_row(expl, _X, ProblemType.Classification, target_class=1)
    c0 = _compute_shap_values_for_row(expl, _X, ProblemType.Classification, target_class=0)

    assert np.allclose(c1, [0.5, -0.2, 0.1])
    assert np.allclose(c0, [-0.5, 0.2, -0.1])  # mirror
    assert np.allclose(c0, -c1)  # explicitly the negation
    assert not np.allclose(c0, c1)  # the original bug: were identical


def test_2d_regression_never_negated():
    vals = np.array([[0.5, -0.2, 0.1]])
    expl = _FakeExplainer(vals)
    out = _compute_shap_values_for_row(expl, _X, ProblemType.Regression, target_class=None)
    assert np.allclose(out, [0.5, -0.2, 0.1])  # untouched


# ---------------------------------------------------------------------------
# 3-D output (newer TreeExplainer, e.g. RandomForest): unchanged, class-aware
# ---------------------------------------------------------------------------


def test_3d_selects_requested_class():
    # (1, n_features, n_classes) with class0 = -class1
    arr = np.array([[[-0.5, 0.5], [0.2, -0.2], [-0.1, 0.1]]])  # (1, 3, 2)
    expl = _FakeExplainer(arr)

    c0 = _compute_shap_values_for_row(expl, _X, ProblemType.Classification, target_class=0)
    c1 = _compute_shap_values_for_row(expl, _X, ProblemType.Classification, target_class=1)

    assert np.allclose(c0, [-0.5, 0.2, -0.1])
    assert np.allclose(c1, [0.5, -0.2, 0.1])
    assert np.allclose(c0, -c1)


# ---------------------------------------------------------------------------
# list output (older TreeExplainer): unchanged, class-aware
# ---------------------------------------------------------------------------


def test_list_output_selects_requested_class():
    out = [np.array([[-0.5, 0.2, -0.1]]), np.array([[0.5, -0.2, 0.1]])]  # [class0, class1]
    expl = _FakeExplainer(out)

    c0 = _compute_shap_values_for_row(expl, _X, ProblemType.Classification, target_class=0)
    c1 = _compute_shap_values_for_row(expl, _X, ProblemType.Classification, target_class=1)

    assert np.allclose(c0, [-0.5, 0.2, -0.1])
    assert np.allclose(c1, [0.5, -0.2, 0.1])


def test_all_paths_return_1d():
    for output in (
        np.array([[0.5, -0.2, 0.1]]),  # 2-D
        np.array([[[-0.5, 0.5], [0.2, -0.2], [-0.1, 0.1]]]),  # 3-D
        [np.array([[-0.5, 0.2, -0.1]]), np.array([[0.5, -0.2, 0.1]])],  # list
    ):
        out = _compute_shap_values_for_row(
            _FakeExplainer(output), _X, ProblemType.Classification, target_class=1
        )
        assert out.ndim == 1 and out.shape == (3,)
