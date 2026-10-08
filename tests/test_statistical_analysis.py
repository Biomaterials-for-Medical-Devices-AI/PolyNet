import numpy as np

from polynet.utils.statistical_analysis import regression_pvalue_matrix


def _accurate_vs_noisy(seed: int = 0, n: int = 200):
    rng = np.random.default_rng(seed)
    y_true = rng.normal(0.0, 10.0, size=n)
    accurate = y_true + rng.normal(0.0, 2.0, size=n)  # unbiased, small error
    noisy = y_true + rng.normal(0.0, 20.0, size=n)  # unbiased, large error
    return y_true, np.vstack([accurate, noisy])


def test_regression_wilcoxon_detects_accuracy_difference():
    y_true, preds = _accurate_vs_noisy()
    p = regression_pvalue_matrix(y_true, preds, test="wilcoxon")
    assert p[0, 1] < 0.05


def test_regression_ttest_detects_accuracy_difference():
    y_true, preds = _accurate_vs_noisy()
    p = regression_pvalue_matrix(y_true, preds, test="ttest")
    assert p[0, 1] < 0.05


def test_regression_pvalue_matrix_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(1)
    y_true = rng.normal(size=100)
    preds = np.vstack([y_true + rng.normal(0, s, size=100) for s in (0.5, 1.0, 2.0)])
    for test in ("wilcoxon", "ttest"):
        p = regression_pvalue_matrix(y_true, preds, test=test)
        np.testing.assert_array_equal(p, p.T)
        np.testing.assert_array_equal(np.diag(p), np.ones(3))


# ---------------------------------------------------------------------------
# Metric comparison across repeated splits (B4)
# ---------------------------------------------------------------------------

import pytest  # noqa: E402
from scipy.stats import ttest_rel, wilcoxon  # noqa: E402

from polynet.utils.statistical_analysis import (  # noqa: E402
    MIN_SPLITS_FOR_WILCOXON,
    mean_split_sizes,
    metrics_pvalue_matrix,
    min_wilcoxon_pvalue,
    nadeau_bengio_ttest,
)


@pytest.mark.parametrize("n", [3, 5, 6, 8])
def test_min_wilcoxon_pvalue_matches_scipy_extreme_case(n):
    """All differences with the same sign is the most extreme outcome."""
    x = np.arange(1, n + 1, dtype=float)
    _, p = wilcoxon(x + 10, x, alternative="two-sided")  # every difference positive
    assert p == pytest.approx(min_wilcoxon_pvalue(n))


def test_wilcoxon_needs_six_splits_for_significance():
    assert min_wilcoxon_pvalue(MIN_SPLITS_FOR_WILCOXON - 1) > 0.05
    assert min_wilcoxon_pvalue(MIN_SPLITS_FOR_WILCOXON) < 0.05


def test_corrected_ttest_reduces_to_paired_ttest_without_correction():
    rng = np.random.default_rng(0)
    x, y = rng.normal(1.0, 0.2, 10), rng.normal(0.9, 0.2, 10)
    assert nadeau_bengio_ttest(x, y, test_train_ratio=0.0) == pytest.approx(ttest_rel(x, y).pvalue)


def test_corrected_ttest_matches_hand_calculation_and_is_more_conservative():
    x = np.array([0.80, 0.82, 0.79, 0.85, 0.81])
    y = np.array([0.75, 0.78, 0.77, 0.80, 0.74])
    d = x - y
    k, ratio = len(d), 0.25
    t_stat = d.mean() / np.sqrt((1 / k + ratio) * d.var(ddof=1))
    from scipy.stats import t as student_t

    expected = 2 * student_t.sf(abs(t_stat), df=k - 1)
    p = nadeau_bengio_ttest(x, y, test_train_ratio=ratio)
    assert p == pytest.approx(expected)
    assert p > ttest_rel(x, y).pvalue


@pytest.mark.parametrize(
    "x, y, expected",
    [([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 1.0), ([2.0, 2.0, 2.0], [1.0, 1.0, 1.0], 0.0)],
)
def test_corrected_ttest_zero_variance(x, y, expected):
    assert nadeau_bengio_ttest(x, y, 0.2) == expected


def test_corrected_ttest_needs_two_splits():
    assert np.isnan(nadeau_bengio_ttest([1.0], [0.5], 0.2))


def test_metrics_matrix_corrected_ttest():
    metrics = {
        "A": [0.80, 0.82, 0.79, 0.85],
        "B": [0.75, 0.78, 0.77, 0.80],
        "C": [0.7, 0.9, 0.8, 0.6],
    }
    p, names = metrics_pvalue_matrix(metrics, test="corrected_ttest", test_train_ratio=0.25)
    assert names == ["A", "B", "C"]
    np.testing.assert_array_equal(p, p.T)
    np.testing.assert_array_equal(np.diag(p), np.ones(3))
    assert p[0, 1] == pytest.approx(nadeau_bengio_ttest(metrics["A"], metrics["B"], 0.25))
    with pytest.raises(ValueError, match="test_train_ratio"):
        metrics_pvalue_matrix(metrics, test="corrected_ttest")


def test_default_metric_test_is_unchanged_wilcoxon():
    metrics = {"A": [0.80, 0.82, 0.79, 0.85, 0.81, 0.83], "B": [0.75, 0.78, 0.77, 0.80, 0.74, 0.79]}
    p, _ = metrics_pvalue_matrix(metrics)
    expected = wilcoxon(metrics["A"], metrics["B"], zero_method="pratt").pvalue
    assert p[0, 1] == pytest.approx(expected)


def test_mean_split_sizes():
    split = {"train": [["a"] * 60, ["a"] * 62], "val": [["b"] * 15] * 2, "test": [["c"] * 20] * 2}
    assert mean_split_sizes(split) == {"Training": 61.0, "Validation": 15.0, "Test": 20.0}
