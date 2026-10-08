"""
tests/test_shap_reproducibility.py
==================================
SHAP's KernelExplainer (used for SVMs) samples randomly and takes no seed.
PolyNet seeds its background sample and each sample's computation, so a
sample's SHAP values do not depend on the run or on which samples were
explained before it.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.svm import SVR

from polynet.config.enums import ProblemType
from polynet.explainability import shap_explain

pytest.importorskip("shap")

_RNG = np.random.default_rng(0)
_X = _RNG.normal(size=(80, 6))
_MODEL = SVR().fit(_X, _X[:, 0] - 2 * _X[:, 1] + _RNG.normal(scale=0.1, size=80))


def _explain(rows):
    explainer = shap_explain._select_shap_explainer(_MODEL, _X)
    return {
        i: shap_explain._compute_shap_values_for_row(explainer, _X[i], ProblemType.Regression, None)
        for i in rows
    }


def test_svm_shap_values_do_not_depend_on_run_or_order():
    alone = _explain([5])[5]
    np.random.seed(123)  # unrelated global randomness before explaining
    after_others = _explain([0, 1, 2, 5])[5]
    assert np.array_equal(alone, after_others)


def test_explaining_does_not_change_the_global_random_state():
    np.random.seed(7)
    expected = np.random.random()
    np.random.seed(7)
    _explain([0])
    assert np.random.random() == expected


def test_shap_cache_from_an_older_version_is_recomputed(tmp_path):
    cache = pd.DataFrame(
        {
            "model_type": ["svm"],
            "iteration": ["1"],
            "sample_id": ["s1"],
            "class_idx": ["0"],
            "f": [0.5],
        }
    )
    path = shap_explain._shap_cache_path(tmp_path, "rdkit")
    path.parent.mkdir(parents=True)
    cache.to_csv(path, index=False)  # written without a version
    assert shap_explain._load_shap_cache(tmp_path, "rdkit").empty

    shap_explain._save_shap_cache(cache, tmp_path, "rdkit")
    loaded = shap_explain._load_shap_cache(tmp_path, "rdkit")
    assert list(loaded.columns) == list(cache.columns) and loaded["f"].tolist() == [0.5]
