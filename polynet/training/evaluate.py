"""
polynet.training.evaluate
==========================
Plotting utilities for visualising training and evaluation results.

Generates and saves learning curves, parity plots, confusion matrices,
and ROC curves based on predictions DataFrames and trained model objects.

Public API
----------
::

    from polynet.training.evaluate import plot_learning_curves, plot_results
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from polynet.config.column_names import get_iterator_name, get_true_label_column_name
from polynet.config.constants import DataSet, ResultColumn
from polynet.config.enums import ProblemType, SplitType

logger = logging.getLogger(__name__)


def plot_learning_curves(models: dict, save_path: Path) -> None:
    """
    Plot and save learning curves for all trained GNN models.

    Parameters
    ----------
    models:
        Dict of ``{model_log_name: model}`` as returned by
        ``train_gnn_ensemble``. Each model is expected to have a
        ``losses`` attribute set to ``(train_losses, val_losses, test_losses)``.
    save_path:
        Directory where plot images will be saved.

    Notes
    -----
    Models whose ``losses`` attribute is ``None`` (e.g. TML models)
    are silently skipped.
    """
    # Deferred import — visualization module is in polynet.visualization
    from polynet.visualization.plots import plot_learning_curve
    from polynet.visualization.utils import save_plot

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        if not hasattr(model, "losses"):
            logger.debug(f"Skipping learning curve for '{model_name}' — no losses attribute.")
            continue

        elif model.losses is None:
            logger.debug(f"Skipping learning curve for '{model_name}' — no losses recorded.")
            continue

        fig = plot_learning_curve(model.losses, title=f"{model_name} Learning Curve")
        out_path = save_path / f"{model_name}_learning_curve.png"
        save_plot(fig=fig, path=out_path)
        logger.info(f"Saved learning curve: {out_path}")


def plot_results(
    predictions: pd.DataFrame,
    split_type: SplitType | str,
    target_variable_name: str,
    ml_algorithms: list[str],
    problem_type: ProblemType | str,
    save_path: Path,
    class_names: dict | None = None,
) -> None:
    """
    Plot and save evaluation charts for all trained models on the test set.

    For classification tasks, generates confusion matrices and ROC curves.
    For regression tasks, generates parity plots.

    Parameters
    ----------
    predictions:
        Predictions DataFrame produced by the inference step.
    split_type:
        The split strategy used — determines the iterator column name.
    target_variable_name:
        Name of the target property.
    ml_algorithms:
        List of model log names in the format ``"{algorithm}_{iteration}"``.
    problem_type:
        Whether this is classification or regression.
    save_path:
        Directory where plot images will be saved.
    class_names:
        Optional mapping from class index to display label (e.g.
        ``{0: "inactive", 1: "active"}``). Used for confusion matrix labels.
    """
    from polynet.visualization.plots import plot_auroc, plot_confusion_matrix, plot_parity
    from polynet.visualization.utils import save_plot

    split_type = SplitType(split_type) if isinstance(split_type, str) else split_type
    problem_type = ProblemType(problem_type) if isinstance(problem_type, str) else problem_type

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    iterator = get_iterator_name(split_type)
    label_col = get_true_label_column_name(target_variable_name)

    for model_log in ml_algorithms:
        algorithm, iteration = model_log.rsplit("_", 1)

        test_df = predictions.loc[
            (predictions[iterator] == iteration) & (predictions[ResultColumn.SET] == DataSet.Test)
        ]

        result_cols = [col for col in test_df.columns if algorithm in col]
        predicted_col = result_cols.pop(0)

        if problem_type == ProblemType.Classification:
            _plot_classification_results(
                test_df=test_df,
                label_col=label_col,
                predicted_col=predicted_col,
                prob_cols=result_cols,
                algorithm=algorithm,
                iteration=iteration,
                target_variable_name=target_variable_name,
                class_names=class_names,
                save_path=save_path,
                plot_auroc_fn=plot_auroc,
                plot_cm_fn=plot_confusion_matrix,
                save_fn=save_plot,
            )

        elif problem_type == ProblemType.Regression:
            fig = plot_parity(
                y_true=test_df[label_col],
                y_pred=test_df[predicted_col],
                title=f"{target_variable_name}\nParity Plot for\n{algorithm} - {iteration}",
            )
            out_path = save_path / f"{algorithm}_{iteration}_parity_plot.png"
            save_plot(fig=fig, path=out_path)
            logger.info(f"Saved parity plot: {out_path}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _plot_classification_results(
    test_df: pd.DataFrame,
    label_col: str,
    predicted_col: str,
    prob_cols: list[str],
    algorithm: str,
    iteration: str,
    target_variable_name: str,
    class_names: dict | None,
    save_path: Path,
    plot_auroc_fn,
    plot_cm_fn,
    save_fn,
) -> None:
    """Save confusion matrix and per-class ROC curves for a classification model."""
    display_labels = list(class_names.values()) if class_names else None

    fig = plot_cm_fn(
        y_true=test_df[label_col],
        y_pred=test_df[predicted_col],
        display_labels=display_labels,
        title=f"{target_variable_name}\nConfusion Matrix for\n{algorithm} - {iteration}",
    )
    cm_path = save_path / f"{algorithm}_{iteration}_confusion_matrix.png"
    save_fn(fig=fig, path=cm_path)
    logger.info(f"Saved confusion matrix: {cm_path}")

    for class_num, prob_col in enumerate(prob_cols):
        # For binary classification with a single probability column,
        # class_num refers to the positive class (class 1)
        display_class = class_num + 1 if len(prob_cols) == 1 else class_num

        fig = plot_auroc_fn(
            y_true=test_df[label_col],
            y_scores=test_df[prob_col],
            title=(
                f"{target_variable_name}\nROC Curve for\n"
                f"{algorithm} Class {display_class} - {iteration}"
            ),
        )
        roc_path = save_path / f"{algorithm}_{iteration}_class_{display_class}_roc_curve.png"
        save_fn(fig=fig, path=roc_path)
        logger.info(f"Saved ROC curve: {roc_path}")
