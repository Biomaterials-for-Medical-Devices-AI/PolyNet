import numpy as np
from scipy.stats import t as student_t
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from polynet.config.constants import DataSet

# Metric-level tests across repeated splits: user-facing label -> ``metrics_pvalue_matrix`` test.
METRIC_COMPARISON_TESTS: dict[str, str] = {
    "Wilcoxon signed-rank": "wilcoxon",
    "Nadeau–Bengio corrected t-test": "corrected_ttest",
}

# Below this many splits, Wilcoxon cannot reach two-sided p < 0.05 (see min_wilcoxon_pvalue).
MIN_SPLITS_FOR_WILCOXON = 6

# User-facing label -> statsmodels ``multipletests`` method name.
# ``None`` leaves the raw (uncorrected) p-values untouched.
MULTIPLE_COMPARISON_METHODS: dict[str, str | None] = {
    "Holm-Bonferroni": "holm",
    "Bonferroni": "bonferroni",
    "Benjamini-Hochberg (FDR)": "fdr_bh",
    "None (raw p-values)": None,
}


def correct_pvalue_matrix(p_matrix: np.ndarray, method: str | None = "holm") -> np.ndarray:
    """
    Apply a multiple-comparison correction to a symmetric pairwise p-value matrix.

    Pairwise model comparison produces ``k·(k - 1)/2`` simultaneous hypotheses,
    so the raw p-values must be adjusted to control the family-wise error rate
    (or the false discovery rate) before significance is judged.

    Only the unique upper-triangle (``i < j``) finite entries are treated as the
    family of hypotheses. Corrected values are mirrored back into the lower
    triangle so the matrix stays symmetric; the diagonal is left untouched.

    Parameters
    ----------
    p_matrix:
        Symmetric ``(n_models, n_models)`` matrix of raw p-values.
    method:
        Any ``statsmodels.stats.multitest.multipletests`` method name
        (e.g. ``"holm"``, ``"bonferroni"``, ``"fdr_bh"``). ``None`` returns the
        matrix unchanged.

    Returns
    -------
    np.ndarray
        A new matrix with corrected p-values (the input is not modified).
    """
    if method is None:
        return p_matrix

    corrected_matrix = np.array(p_matrix, dtype=float)
    n = corrected_matrix.shape[0]
    upper = np.triu_indices(n, k=1)
    pvals = corrected_matrix[upper]

    finite = np.isfinite(pvals)
    if finite.sum() == 0:
        return corrected_matrix

    adjusted = pvals.copy()
    adjusted[finite] = multipletests(pvals[finite], method=method)[1]

    corrected_matrix[upper] = adjusted
    corrected_matrix[(upper[1], upper[0])] = adjusted  # mirror to lower triangle
    return corrected_matrix


def min_wilcoxon_pvalue(n_splits: int) -> float:
    """
    Smallest two-sided p-value an exact Wilcoxon signed-rank test can return.

    With ``n`` paired values, the most extreme outcome (all differences with
    the same sign) has probability ``2 / 2**n``. For ``n = 5`` this is 0.0625,
    so no comparison over five or fewer splits can be significant at 0.05.

    Parameters
    ----------
    n_splits:
        Number of paired metric values (repeated splits).

    Returns
    -------
    float
        The smallest achievable two-sided p-value (capped at 1).
    """
    return min(1.0, 2.0 / 2**n_splits)


def mean_split_sizes(split_indices: dict) -> dict[str, float]:
    """
    Average number of samples per set across the repeated splits.

    Parameters
    ----------
    split_indices:
        Contents of ``split_indices.json``: ``{"train": [...], "val": [...],
        "test": [...]}``, each a list with one list of sample IDs per split.

    Returns
    -------
    dict[str, float]
        ``{"Training": …, "Validation": …, "Test": …}`` (``DataSet`` labels).
    """
    labels = {"train": DataSet.Training, "val": DataSet.Validation, "test": DataSet.Test}
    return {
        labels[key]: float(np.mean([len(ids) for ids in split_indices[key]]))
        for key in labels
        if split_indices.get(key)
    }


def nadeau_bengio_ttest(x, y, test_train_ratio: float) -> float:
    """
    Nadeau–Bengio corrected resampled t-test for two models' metrics.

    The metrics come from ``k`` repeated random train/test splits of the same
    dataset. Because the training sets overlap, the ``k`` differences are not
    independent and a plain paired t-test underestimates their variance. The
    correction of Nadeau & Bengio (Machine Learning 52, 239–281, 2003) replaces
    ``var / k`` by ``(1/k + n_test/n_train) * var``.

    Parameters
    ----------
    x, y:
        Metric values of the two models, paired by split (length ``k >= 2``).
    test_train_ratio:
        ``n_test / n_train`` — size of the evaluated set over the size of the
        training set. ``0`` gives the ordinary paired t-test.

    Returns
    -------
    float
        Two-sided p-value (Student t with ``k - 1`` degrees of freedom).
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    k = len(d)
    if k < 2:
        return np.nan
    mean, var = d.mean(), d.var(ddof=1)
    if var == 0:
        return 1.0 if mean == 0 else 0.0
    t_stat = mean / np.sqrt((1.0 / k + test_train_ratio) * var)
    return float(2 * student_t.sf(abs(t_stat), df=k - 1))


def metrics_pvalue_matrix(metrics_dict, test="wilcoxon", test_train_ratio: float | None = None):
    """
    Compare models pairwise on a metric measured over repeated splits.

    Parameters
    ----------
    metrics_dict:
        ``{model_name: 1D array-like of metric values, one per split}``. Values
        are paired by position (split).
    test:
        ``"wilcoxon"`` (default, Wilcoxon signed-rank), ``"ttest"`` (paired
        t-test) or ``"corrected_ttest"`` (Nadeau–Bengio corrected resampled
        t-test, which accounts for the overlap between repeated splits).
    test_train_ratio:
        ``n_test / n_train``; required for ``"corrected_ttest"``.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        ``(p_matrix, model_names)`` — symmetric p-values with ones on the
        diagonal.

    Notes
    -----
    With ``k`` splits Wilcoxon cannot return a two-sided p-value below
    ``2 / 2**k`` (see ``min_wilcoxon_pvalue``): at least 6 splits are needed
    for p < 0.05.
    """
    if test == "corrected_ttest" and test_train_ratio is None:
        raise ValueError("test_train_ratio (n_test / n_train) is required for 'corrected_ttest'.")
    model_names = list(metrics_dict.keys())
    n = len(model_names)
    p_matrix = np.ones((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            x = np.asarray(metrics_dict[model_names[i]], dtype=float)
            y = np.asarray(metrics_dict[model_names[j]], dtype=float)

            # align lengths (if needed) and drop NaNs
            L = min(len(x), len(y))
            x, y = x[:L], y[:L]
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]

            if len(x) == 0:
                pij = np.nan
            else:
                if test == "wilcoxon":
                    # 'pratt' handles zeros in differences gracefully
                    _, pij = wilcoxon(x, y, zero_method="pratt", alternative="two-sided")
                elif test == "ttest":
                    _, pij = ttest_rel(x, y, nan_policy="omit")
                elif test == "corrected_ttest":
                    pij = nadeau_bengio_ttest(x, y, test_train_ratio)
                else:
                    raise ValueError("test must be 'wilcoxon', 'ttest' or 'corrected_ttest'")

            p_matrix[i, j] = p_matrix[j, i] = pij

    return p_matrix, model_names


def regression_pvalue_matrix(y_true, predictions, test="wilcoxon"):
    """
    Compare regression models pairwise using paired tests on absolute errors.

    For each sample the absolute error ``|y_true - y_pred|`` of each model is
    computed, and the paired absolute errors of every pair of models are
    compared. This tests for a difference in accuracy; comparing signed
    residuals instead would only test for a difference in bias, so two
    unbiased models of very different accuracy would look alike.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground truth values.
    predictions : np.ndarray of shape (n_models, n_samples)
        Predictions from each model.
    test : str
        Which paired test to apply to the absolute errors: 'wilcoxon'
        (default, Wilcoxon signed-rank, non-parametric) or 'ttest'
        (paired t-test).

    Returns
    -------
    p_matrix : np.ndarray of shape (n_models, n_models)
        Symmetric matrix of p-values for pairwise model comparisons, with
        ones on the diagonal.
    """
    if test not in ("wilcoxon", "ttest"):
        raise ValueError("test must be 'wilcoxon' or 'ttest'")

    y_true = np.asarray(y_true, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    n_models = predictions.shape[0]
    p_matrix = np.ones((n_models, n_models))

    for i in range(n_models):
        for j in range(i + 1, n_models):
            # absolute errors for each model
            e1 = np.abs(y_true - predictions[i])
            e2 = np.abs(y_true - predictions[j])

            if test == "wilcoxon":
                _, p = wilcoxon(e1, e2)
            else:
                _, p = ttest_rel(e1, e2)

            p_matrix[i, j] = p_matrix[j, i] = p

    return p_matrix


def mcnemar_pvalue_matrix(y_true, predictions):
    """
    y_true: np.array of shape (n_samples,)
    predictions: np.array of shape (n_models, n_samples)
    """
    n_models = predictions.shape[0]
    p_matrix = np.ones((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                b = np.sum((predictions[i] == y_true) & (predictions[j] != y_true))
                c = np.sum((predictions[i] != y_true) & (predictions[j] == y_true))
                table = [[0, b], [c, 0]]
                result = mcnemar(table, exact=True)
                p_matrix[i, j] = result.pvalue

    return p_matrix


def significance_marker(p) -> str:
    """Return a significance marker string for a given p-value.

    Args:
        p: The p-value to evaluate.

    Returns:
        str: ``"***"`` (p < 0.001), ``"**"`` (p < 0.01), ``"*"`` (p < 0.05),
        or ``""`` (not significant).
    """
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""
