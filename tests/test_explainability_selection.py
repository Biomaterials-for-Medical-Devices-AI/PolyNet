"""
tests/test_explainability_selection.py
======================================
Which models and molecules the explainability stages explain:

- ``bootstraps`` counts splits from 1, like the model names, so ``"all"``
  must include the last split and ``[1]`` must select ``*_1``;
- requested molecule IDs (strings from configs / split files) must match
  graphs whose ID column is numeric;
- unquoted numeric IDs in YAML are accepted.
"""

import logging
import types

import pytest

from polynet.config.schemas.explainability import ExplainabilityConfig
from polynet.config.schemas.tml_explainability import TMLExplainabilityConfig
from polynet.explainability.selection import match_dataset_ids, select_splits, split_index_of_model

MODELS = ["GCN_1", "GCN_2", "GCN_3"]


def _chosen(bootstraps, n_splits=3):
    splits = select_splits(bootstraps, n_splits)
    return [m for m in MODELS if split_index_of_model(m) in splits]


@pytest.mark.parametrize(
    "bootstraps, expected",
    [("all", MODELS), ([1], ["GCN_1"]), ([3], ["GCN_3"]), ([1, 3], ["GCN_1", "GCN_3"])],
)
def test_bootstraps_select_the_right_models(bootstraps, expected):
    assert _chosen(bootstraps) == expected


def test_out_of_range_bootstraps_are_skipped_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        assert _chosen([2, 0, 4]) == ["GCN_2"]
    assert "[0, 4] are out of range (splits are numbered 1–3)" in caplog.text


def test_string_ids_match_numeric_graph_ids(caplog):
    dataset = [types.SimpleNamespace(idx=i) for i in (0, 7, 12)]
    with caplog.at_level(logging.WARNING):
        matched = match_dataset_ids(dataset, ["12", "0", "99"], "molecule")
    assert matched == [12, 0] and all(isinstance(i, int) for i in matched)
    assert "1 molecule ID(s) not found" in caplog.text


def test_unquoted_numeric_ids_are_accepted():
    assert ExplainabilityConfig(local_explain_mol_ids=[0, 12]).local_explain_mol_ids == ["0", "12"]
    tml_cfg = TMLExplainabilityConfig(local_explain_sample_ids=[3, "a7"])
    assert tml_cfg.local_explain_sample_ids == ["3", "a7"]


# ---------------------------------------------------------------------------
# Global explanations: each model explains only its own split's samples
# ---------------------------------------------------------------------------

import pandas as pd  # noqa: E402

from polynet.explainability.explain import build_display_data  # noqa: E402
from polynet.explainability.selection import (  # noqa: E402
    samples_per_model,
    samples_per_model_from_predictions,
)
from polynet.explainability.shap_explain import merge_shap_attributions  # noqa: E402

# split 1: train a,b  val c  test d,e   |   split 2: train d,e  val a  test b,c
SPLITS = ([["a", "b"], ["d", "e"]], [["c"], ["a"]], [["d", "e"], ["b", "c"]])


def test_samples_per_model_uses_each_models_own_split():
    allowed = samples_per_model(["GCN_1", "GCN_2", "rf-morgan_2"], SPLITS, "test")
    assert allowed == {"GCN_1": {"d", "e"}, "GCN_2": {"b", "c"}, "rf-morgan_2": {"b", "c"}}
    assert samples_per_model(["GCN_2"], SPLITS, "all") == {"GCN_2": {"a", "b", "c", "d", "e"}}


def test_samples_per_model_from_gui_predictions():
    preds = pd.DataFrame(
        {"Set": ["Training", "Test", "Test", "Training"], "iteration": [1, 1, 2, 2]},
        index=["a", "b", "a", "b"],
    )
    by_model = samples_per_model_from_predictions(preds, ["GCN_1", "GCN_2"], "iteration", "Test")
    assert by_model == {"GCN_1": {"b"}, "GCN_2": {"a"}}
    assert samples_per_model_from_predictions(preds, ["GCN_1"], "iteration", "All") is None
    assert samples_per_model_from_predictions(preds, ["GCN_1"], None, "Test") is None


def test_display_data_keeps_each_models_own_molecules():
    cache = {"GCN": {"1": {"a": 1, "b": 2}, "2": {"a": 3, "b": 4}}}
    models = {"GCN_1": None, "GCN_2": None}
    everything = build_display_data(cache, models, ["a", "b"])
    assert everything == {"GCN": {"1": {"a": 1, "b": 2}, "2": {"a": 3, "b": 4}}}
    own = build_display_data(cache, models, ["a", "b"], {"GCN_1": {"b"}, "GCN_2": {"a"}})
    assert own == {"GCN": {"1": {"b": 2}, "2": {"a": 3}}}


def _shap_cache(rows):
    return pd.DataFrame(rows, columns=["model_type", "iteration", "sample_id", "class_idx", "f1"])


def test_shap_merge_uses_exact_model_instances():
    # rf_1 and xgb_2 selected; cached rf_2 / xgb_1 rows must not leak in.
    cache = _shap_cache(
        [
            ("rf", "1", "a", 0, 1.0),
            ("rf", "2", "a", 0, 100.0),
            ("xgb", "1", "a", 0, 100.0),
            ("xgb", "2", "a", 0, 3.0),
        ]
    )
    merged = merge_shap_attributions(cache, ["rf-morgan_1", "xgb-morgan_2"], ["a"])
    assert merged.loc["a", "f1"] == 2.0  # mean of 1.0 and 3.0 only


def test_shap_merge_restricts_each_model_to_its_samples():
    cache = _shap_cache(
        [("rf", "1", "a", 0, 1.0), ("rf", "1", "b", 0, 5.0), ("rf", "2", "a", 0, 9.0)]
    )
    merged = merge_shap_attributions(
        cache,
        ["rf-morgan_1", "rf-morgan_2"],
        ["a", "b"],
        {"rf-morgan_1": {"b"}, "rf-morgan_2": {"a"}},
    )
    assert merged["f1"].to_dict() == {"a": 9.0, "b": 5.0}


def test_tml_validation_warning_covers_cli_and_gui_set_names():
    from polynet.config.constants import DataSet
    from polynet.explainability.selection import tml_explain_set_includes_validation

    for name in ("validation", "all", DataSet.Validation, "All"):
        assert tml_explain_set_includes_validation(name)
    for name in ("test", "train", DataSet.Test, DataSet.Training, "External", None):
        assert not tml_explain_set_includes_validation(name)
