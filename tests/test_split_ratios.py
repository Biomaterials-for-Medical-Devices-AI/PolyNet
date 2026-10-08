"""
tests/test_split_ratios.py
==========================
Train / validation / test splits are drawn with astartes (``sampler="random"``).
``test_ratio`` and ``val_ratio`` are fractions of the full dataset: test 0.1 +
validation 0.1 gives an 80/10/10 split. astartes rounds the training and
validation sizes down and gives the remaining samples to the test set.
"""

import logging
import math

import pandas as pd
import pytest

from polynet.factories.dataloader import get_data_split_indices


def _split(n, val_ratio, test_ratio, balance=None, y=None, n_iter=3, method="random"):
    data = pd.DataFrame({"y": y if y is not None else range(n)}, index=range(n))
    logging.disable(logging.INFO)
    try:
        return data, get_data_split_indices(
            data=data,
            split_type="train_val_test",
            n_bootstrap_iterations=n_iter,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            target_variable_col="y",
            split_method=method,
            train_set_balance=balance,
            random_seed=0,
        )
    finally:
        logging.disable(logging.NOTSET)


def _sizes(*args, **kwargs):
    _, (train, val, test) = _split(*args, **kwargs)
    return [(len(a), len(b), len(c)) for a, b, c in zip(train, val, test)]


@pytest.mark.parametrize(
    "n, val_ratio, test_ratio, expected",
    [
        (100, 0.1, 0.1, (80, 10, 10)),
        (200, 0.15, 0.15, (140, 30, 30)),
        (50, 0.2, 0.2, (30, 10, 10)),
        (1000, 0.1, 0.2, (700, 100, 200)),
    ],
)
def test_ratios_are_fractions_of_the_full_dataset(n, val_ratio, test_ratio, expected):
    assert set(_sizes(n, val_ratio, test_ratio)) == {expected}


@pytest.mark.parametrize("n", [37, 83, 151])
def test_train_and_validation_round_down_and_test_takes_the_rest(n):
    for n_train, n_val, n_test in _sizes(n, 0.15, 0.15):
        assert n_val == math.floor(0.15 * n)
        assert n_train + n_val + n_test == n
        assert abs(n_train - 0.7 * n) <= 1


def test_splits_are_disjoint_and_differ_per_iteration():
    _, (train, val, test) = _split(60, 0.2, 0.2)
    for a, b, c in zip(train, val, test):
        assert not (set(a) & set(b) or set(a) & set(c) or set(b) & set(c))
    assert set(test[0]) != set(test[1])


def test_stratified_split_keeps_class_proportions():
    y = [1] * 40 + [0] * 160  # 20 % minority
    data, (train, val, test) = _split(200, 0.1, 0.1, y=y, method="stratified")
    for part in (train[0], val[0], test[0]):
        assert data.loc[part, "y"].mean() == pytest.approx(0.2, abs=0.01)


def test_stratified_split_rejects_classes_too_small_for_three_sets():
    with pytest.raises(ValueError, match=r"class 1 \(3 sample"):
        _split(50, 0.2, 0.2, y=[1] * 3 + [0] * 47, method="stratified")


def test_balancing_keeps_the_train_to_validation_proportion_with_stratified_splits():
    # 20 % minority class; train and validation are each balanced after the split.
    y = [1] * 40 + [0] * 160
    for n_train, n_val, _ in _sizes(200, 0.1, 0.1, balance=0.5, y=y, method="stratified"):
        assert n_val / (n_train + n_val) == pytest.approx(0.1 / 0.9, abs=0.01)
