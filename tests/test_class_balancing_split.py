"""
tests/test_class_balancing_split.py
===================================
Class balancing in TrainValTest splits follows ACS Appl. Mater. Interfaces
2023, 15 (11), 14155–14163: the non-test data is balanced *before* the
validation split, so training and validation are balanced and only the test
set keeps the original class distribution.
"""

import logging

import pandas as pd
import pytest

from polynet.factories.dataloader import get_data_split_indices

# 20 % minority class.
_DF = pd.DataFrame({"y": [1] * 40 + [0] * 160}, index=[f"s{i}" for i in range(200)])


def _split(method="stratified", balance=0.5, seed=42, n_iter=1):
    logging.disable(logging.INFO)
    try:
        return get_data_split_indices(
            data=_DF,
            split_type="train_val_test",
            n_bootstrap_iterations=n_iter,
            val_ratio=0.15,
            test_ratio=0.15,
            target_variable_col="y",
            split_method=method,
            train_set_balance=balance,
            random_seed=seed,
        )
    finally:
        logging.disable(logging.NOTSET)


def _minority(idx):
    return _DF.loc[idx, "y"].mean()


def test_train_and_val_balanced_test_keeps_original_distribution():
    train, val, test = _split()
    assert _minority(train[0]) == pytest.approx(0.5, abs=0.07)
    assert _minority(val[0]) == pytest.approx(0.5, abs=0.07)
    assert _minority(test[0]) == pytest.approx(0.2, abs=0.01)  # stratified test = original


def test_only_majority_samples_are_dropped_and_splits_are_disjoint():
    train, val, test = _split()
    used = set(train[0]) | set(val[0]) | set(test[0])
    assert len(used) == len(train[0]) + len(val[0]) + len(test[0])  # disjoint
    dropped = set(_DF.index) - used
    assert dropped and all(_DF.loc[i, "y"] == 0 for i in dropped)


def test_balancing_is_deterministic_per_seed():
    a, b = _split(n_iter=2), _split(n_iter=2)
    for part_a, part_b in zip(a, b):
        assert [list(x) for x in part_a] == [list(x) for x in part_b]


def test_no_balancing_keeps_all_samples():
    train, val, test = _split(balance=None)
    assert len(train[0]) + len(val[0]) + len(test[0]) == len(_DF)
