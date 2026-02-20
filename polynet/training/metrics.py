"""
polynet.training.metrics
=========================
Evaluation metric calculation for GNN and TML polymer property models.

Covers both classification (accuracy, precision, recall, specificity,
AUROC, MCC, F1, G-mean) and regression (RMSE, R², MAE) metrics.

Also provides ``compute_class_weights`` for imbalanced classification,
which previously lived in ``polynet.factories``.

Public API
----------
::

    from polynet.training.metrics import calculate_metrics, get_metrics, compute_class_weights
"""

from __future__ import annotations

import logging
from math import sqrt

import numpy as np
import pandas as pd
import torch

# from imblearn.metrics import geometric_mean_score as gmean
# from imblearn.metrics import specificity_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import matthews_corrcoef as mcc

from polynet.config.column_names import get_iterator_name, get_true_label_column_name
from polynet.config.constants import ResultColumn
from polynet.config.enums import EvaluationMetric, ProblemType, SplitType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Class weight computation
# ---------------------------------------------------------------------------


def compute_class_weights(
    labels: list[int], num_classes: int, imbalance_strength: float = 1.0
) -> torch.Tensor:
    """
    Compute class weights for imbalanced classification datasets.

    Interpolates between no correction (uniform weights proportional to
    class frequency) and full inverse-frequency correction based on
    ``imbalance_strength``.

    Parameters
    ----------
    labels:
        Integer class labels for all training samples.
    num_classes:
        Total number of classes.
    imbalance_strength:
        Interpolation factor between no correction (``0.0``) and full
        inverse-frequency correction (``1.0``). Values between 0 and 1
        apply partial correction.

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``(num_classes,)`` suitable for passing
        to ``nn.CrossEntropyLoss(weight=...)``.

    Examples
    --------
    >>> weights = compute_class_weights([0, 0, 0, 1], num_classes=2, imbalance_strength=1.0)
    >>> loss_fn = nn.CrossEntropyLoss(weight=weights)
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    total = counts.sum()

    freq_weights = counts / total
    inverse_weights = 1.0 / (counts + 1e-8)
    inverse_weights /= inverse_weights.sum()

    weights = (1 - imbalance_strength) * freq_weights + imbalance_strength * inverse_weights
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------------------------


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    problem_type: ProblemType | str,
    y_probs: pd.DataFrame | np.ndarray | None = None,
) -> dict[EvaluationMetric, float]:
    """
    Calculate evaluation metrics for a single set of predictions.

    Parameters
    ----------
    y_true:
        Ground-truth labels or values.
    y_pred:
        Predicted labels or values.
    problem_type:
        Whether this is a classification or regression task.
    y_probs:
        Class probability scores (required for AUROC in classification).
        Shape ``(n_samples, n_classes)`` or ``(n_samples,)`` for binary.
        Ignored for regression.

    Returns
    -------
    dict[EvaluationMetric, float]
        Metric name → value mapping. AUROC is ``None`` if ``y_probs``
        is not provided for a classification task.
    """
    problem_type = ProblemType(problem_type) if isinstance(problem_type, str) else problem_type

    if problem_type == ProblemType.Classification:
        return {
            EvaluationMetric.Accuracy: accuracy_score(y_true, y_pred),
            EvaluationMetric.Precision: precision_score(y_true, y_pred),
            EvaluationMetric.Recall: recall_score(y_true, y_pred),
            # EvaluationMetric.Specificity: specificity_score(y_true, y_pred),
            EvaluationMetric.AUROC: (
                roc_auc_score(y_true=y_true, y_score=y_probs) if y_probs is not None else None
            ),
            EvaluationMetric.MCC: mcc(y_true, y_pred),
            EvaluationMetric.F1Score: f1_score(y_true, y_pred),
            # EvaluationMetric.GScore: gmean(y_true, y_pred),
        }

    return {
        EvaluationMetric.RMSE: sqrt(mean_squared_error(y_true, y_pred)),
        EvaluationMetric.R2: r2_score(y_true, y_pred),
        EvaluationMetric.MAE: mean_absolute_error(y_true, y_pred),
    }


def get_metrics(
    predictions: pd.DataFrame,
    split_type: SplitType | str,
    target_variable_name: str,
    trained_models: list[str],
    problem_type: ProblemType | str,
) -> dict:
    """
    Compute metrics for all trained models across all dataset splits.

    Parses the predictions DataFrame produced by the pipeline and
    computes per-iteration, per-model, per-set metrics.

    Parameters
    ----------
    predictions:
        Predictions DataFrame produced by the inference step. Expected
        to contain an iterator column, a set column, a true label column,
        and one or more prediction columns per model.
    split_type:
        The split strategy used — determines the iterator column name.
    target_variable_name:
        Name of the target property (used to look up the label column).
    trained_models:
        List of model log names in the format ``"{algorithm}_{iteration}"``.
    problem_type:
        Whether this is classification or regression.

    Returns
    -------
    dict
        Nested dict of structure
        ``{iteration: {algorithm: {set: {metric: value}}}}``.
    """
    split_type = SplitType(split_type) if isinstance(split_type, str) else split_type
    problem_type = ProblemType(problem_type) if isinstance(problem_type, str) else problem_type

    iterator = get_iterator_name(split_type)
    label_col = get_true_label_column_name(target_variable_name)

    metrics: dict = {}

    for model_log in trained_models:
        input(model_log)
        algorithm, iteration = model_log.rsplit("_", 1)
        input(f"{algorithm}\n{iteration}")

        if iteration not in metrics:
            metrics[iteration] = {}

        iteration_df = predictions.loc[predictions[iterator] == iteration]
        result_cols = [col for col in iteration_df.columns if algorithm in col]
        predicted_col = result_cols.pop(0)

        metrics[iteration][algorithm] = {}

        for dataset_split in iteration_df[ResultColumn.SET].unique():
            split_df = iteration_df.loc[iteration_df[ResultColumn.SET] == dataset_split]

            metrics[iteration][algorithm][dataset_split] = calculate_metrics(
                y_true=split_df[label_col],
                y_pred=split_df[predicted_col],
                y_probs=(
                    split_df[result_cols] if problem_type == ProblemType.Classification else None
                ),
                problem_type=problem_type,
            )

    return metrics
