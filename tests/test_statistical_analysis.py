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
