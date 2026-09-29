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
from polynet.explainability.selection import (
    match_dataset_ids,
    select_splits,
    split_index_of_model,
)

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
