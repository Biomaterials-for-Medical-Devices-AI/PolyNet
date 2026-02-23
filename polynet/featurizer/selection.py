"""
polynet.featurizer.selection
=============================
Feature selection utilities for descriptor-based polymer representations.

Provides three feature selection strategies:

1. **Correlation filtering** — remove features with Pearson correlation
   above a threshold (``uncorrelated_features``).
2. **Diversity filtering** — remove features with low information entropy
   (``diversity_filter``).
3. **Sequential Forward Selection** — greedily add features one at a time
   based on cross-validated MCC (``sequential_forward_selection``).

Note
----
``sequential_forward_selection`` is a legacy method retained for
backward compatibility. For new workflows, consider using sklearn's
``SequentialFeatureSelector`` which is more efficient and better
integrated with the sklearn ecosystem.

Public API
----------
::

    from polynet.featurizer.selection import (
        uncorrelated_features,
        diversity_filter,
        sequential_forward_selection,
    )
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Correlation filtering
# ---------------------------------------------------------------------------


def uncorrelated_features(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """
    Remove features with Pearson correlation above ``threshold``.

    Iterates through columns in order. A column is kept if none of the
    already-kept columns have a correlation with it above ``threshold``.

    Parameters
    ----------
    df:
        DataFrame of features (no target column).
    threshold:
        Correlation threshold. Features with correlation higher than
        this value with any previously accepted feature are removed.
        Must be in ``(0, 1]``.

    Returns
    -------
    pd.DataFrame
        Subset of ``df`` with correlated columns removed.

    Examples
    --------
    >>> from polynet.featurizer.selection import uncorrelated_features
    >>> filtered = uncorrelated_features(descriptors_df, threshold=0.9)
    """
    corr = df.corr().abs()
    keep: list[str] = []

    for i in range(len(corr.columns)):
        col = corr.columns[i]
        above = corr.iloc[:i, i]
        if keep:
            above = above[keep]
        if len(above[above < threshold]) == len(above):
            keep.append(col)

    logger.info(
        f"Correlation filtering (threshold={threshold}): "
        f"{df.shape[1]} → {len(keep)} features retained."
    )
    return df[keep]


# ---------------------------------------------------------------------------
# Diversity filtering
# ---------------------------------------------------------------------------


def diversity_filter(df: pd.DataFrame, diversity_threshold: float = 0.9) -> pd.DataFrame:
    """
    Remove features with normalised entropy below ``diversity_threshold``.

    Low-entropy features carry little information (e.g. near-constant
    columns). Diversity is computed as the ratio of the column's entropy
    to the maximum possible entropy for that number of unique values.

    Parameters
    ----------
    df:
        DataFrame of features (no target column).
    diversity_threshold:
        Minimum normalised entropy to retain a feature. Must be in
        ``(0, 1]``. Features below this threshold are dropped.

    Returns
    -------
    pd.DataFrame
        Subset of ``df`` with low-diversity columns removed.

    Examples
    --------
    >>> from polynet.featurizer.selection import diversity_filter
    >>> filtered = diversity_filter(descriptors_df, diversity_threshold=0.9)
    """
    diversities: list[float] = []

    for col in df.columns:
        unique_vals = np.unique(df[col])
        n_unique = len(unique_vals)

        if n_unique <= 1:
            diversities.append(0.0)
            continue

        counts = df[col].value_counts(normalize=True).values
        uniform = np.full(n_unique, 1 / n_unique)

        col_entropy = entropy(counts, base=2)
        max_entropy = entropy(uniform, base=2)

        diversity = col_entropy / max_entropy if max_entropy > 0 else 0.0
        diversities.append(round(diversity, 3))

    diversity_series = pd.Series(diversities, index=df.columns)
    to_drop = diversity_series[diversity_series < diversity_threshold].index.tolist()

    logger.info(
        f"Diversity filtering (threshold={diversity_threshold}): "
        f"dropping {len(to_drop)} low-diversity features, "
        f"{df.shape[1] - len(to_drop)} retained."
    )
    return df.drop(columns=to_drop)


# ---------------------------------------------------------------------------
# Sequential Forward Selection (legacy)
# ---------------------------------------------------------------------------


def sequential_forward_selection(
    estimator,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    val_size: float,
    max_features: int,
    cv_folds: int,
) -> tuple[list[float], list[list[float]], list[float], list[float], list[float], list[str]]:
    """
    Greedy Sequential Forward Feature Selection using cross-validated MCC.

    .. deprecated::
        This is a legacy method retained for backward compatibility.
        For new workflows, consider ``sklearn.feature_selection.SequentialFeatureSelector``
        which is more efficient and better integrated with the sklearn ecosystem.

    At each step, the feature whose addition produces the highest mean
    cross-validated MCC is added to the selected feature set.

    Parameters
    ----------
    estimator:
        A fitted or unfitted sklearn-compatible classifier.
    X_train:
        Training feature DataFrame.
    X_test:
        Test feature DataFrame.
    y_train:
        Training target array.
    y_test:
        Test target array.
    val_size:
        Validation fraction for cross-validation within the training set.
    max_features:
        Maximum number of features to select. Must be ≤ ``X_train.shape[1]``.
    cv_folds:
        Number of cross-validation folds.

    Returns
    -------
    tuple
        ``(train_scores, cv_scores, test_scores, sensitivities, specificities, selected_features)``

        - ``train_scores``: MCC on training set at each step.
        - ``cv_scores``: Cross-validated MCC scores (list of lists) at each step.
        - ``test_scores``: MCC on test set at each step.
        - ``sensitivities``: Sensitivity on test set at each step.
        - ``specificities``: Specificity on test set at each step.
        - ``selected_features``: Feature names in the order they were selected.
    """
    import warnings

    warnings.warn(
        "sequential_forward_selection is a legacy method. "
        "Consider using sklearn.feature_selection.SequentialFeatureSelector instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    scaler = StandardScaler()
    selected_features: list[str] = []
    remaining_cols = list(X_train.columns)

    train_scores: list[float] = []
    cv_scores: list[list[float]] = []
    test_scores: list[float] = []
    sensitivities: list[float] = []
    specificities: list[float] = []

    for _ in range(1, max_features):
        candidate_scores: list[float] = []
        candidates = [c for c in remaining_cols if c not in selected_features]

        for col in candidates:
            trial_features = selected_features + [col]
            X_trial = scaler.fit_transform(X_train[trial_features])
            estimator.fit(X_trial, y_train)
            fold_scores = _cross_val_mcc(estimator, X_trial, y_train, cv_folds)
            candidate_scores.append(np.mean(fold_scores))

        best_idx = int(np.argmax(candidate_scores))
        selected_features.append(candidates[best_idx])

        X_sel = scaler.fit_transform(X_train[selected_features])
        X_test_sel = scaler.transform(X_test[selected_features].values)
        estimator.fit(X_sel, y_train)

        train_scores.append(mcc(y_train, estimator.predict(X_sel)))
        cv_scores.append(_cross_val_mcc(estimator, X_sel, y_train, cv_folds))

        estimator.fit(X_sel, y_train)
        y_pred = estimator.predict(X_test_sel)
        confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
        tn, fp, fn, tp = confmat.ravel()

        test_scores.append(mcc(y_test, y_pred))
        sensitivities.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    return (train_scores, cv_scores, test_scores, sensitivities, specificities, selected_features)


def _cross_val_mcc(estimator, X: np.ndarray, y: np.ndarray, n_folds: int) -> list[float]:
    """Compute cross-validated MCC scores using StratifiedKFold."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True)
    scores: list[float] = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        estimator.fit(X_tr, y_tr)
        scores.append(mcc(y_val, estimator.predict(X_val)))

    return scores
