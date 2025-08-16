import numpy as np
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel
from statsmodels.stats.contingency_tables import mcnemar


def regression_pvalue_matrix(y_true, predictions, test="wilcoxon"):
    """
    Compare regression models pairwise using statistical tests on residuals.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground truth values.
    predictions : np.ndarray of shape (n_models, n_samples)
        Predictions from each model.
    test : str
        Which test to use: 'wilcoxon' (default, non-parametric) or 'ttest'.

    Returns
    -------
    p_matrix : np.ndarray of shape (n_models, n_models)
        Matrix of p-values for pairwise model comparisons.
    """

    n_models = predictions.shape[0]
    p_matrix = np.ones((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                # residuals (errors) for each model
                e1 = y_true - predictions[i]
                e2 = y_true - predictions[j]

                if test == "wilcoxon":
                    stat, p = wilcoxon(e1, e2)
                elif test == "ttest":
                    stat, p = ttest_rel(e1, e2)
                else:
                    raise ValueError("test must be 'wilcoxon' or 'ttest'")

                p_matrix[i, j] = p

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
