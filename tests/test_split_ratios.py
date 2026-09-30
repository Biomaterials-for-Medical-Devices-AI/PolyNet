"""
tests/test_split_ratios.py
==========================
``test_ratio`` and ``val_ratio`` are fractions of the full dataset:
test 0.1 + validation 0.1 gives an 80/10/10 split.
"""

import logging

import pandas as pd
import pytest

from polynet.factories.dataloader import get_data_split_indices


def _sizes(n, val_ratio, test_ratio, balance=None, y=None, n_iter=3):
    data = pd.DataFrame({"y": y if y is not None else range(n)}, index=range(n))
    logging.disable(logging.INFO)
    try:
        train, val, test = get_data_split_indices(
            data=data,
            split_type="train_val_test",
            n_bootstrap_iterations=n_iter,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            target_variable_col="y",
            split_method="random",
            train_set_balance=balance,
            random_seed=0,
        )
    finally:
        logging.disable(logging.NOTSET)
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
def test_sizes_are_within_one_sample_of_the_requested_ratios(n):
    for n_train, n_val, n_test in _sizes(n, 0.15, 0.15):
        assert n_train + n_val + n_test == n
        assert abs(n_val - 0.15 * n) <= 1
        assert abs(n_test - 0.15 * n) <= 1


def test_balancing_keeps_the_train_to_validation_proportion():
    # 20 % minority class; balancing drops majority samples from train + val.
    y = [1] * 40 + [0] * 160
    for n_train, n_val, _ in _sizes(200, 0.1, 0.1, balance=0.5, y=y):
        assert n_val / (n_train + n_val) == pytest.approx(0.1 / 0.9, abs=0.02)
