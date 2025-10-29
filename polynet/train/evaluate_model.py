from pathlib import Path
from math import sqrt
import pandas as pd
from polynet.options.enums import SplitTypes, ProblemTypes, Results, EvaluationMetrics, DataSets
from polynet.options.col_names import get_iterator_name, get_true_label_column_name
from imblearn.metrics import geometric_mean_score as gmean, specificity_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef as mcc,
)
from polynet.utils.plot_utils import plot_auroc, plot_confusion_matrix, plot_parity
from polynet.utils import save_plot


def get_metrics(
    predictions: pd.DataFrame,
    split_type: SplitTypes,
    target_variable_name: str,
    trained_models: list,
    problem_type: ProblemTypes,
):
    iterator = get_iterator_name(split_type)

    label_col_name = get_true_label_column_name(target_variable_name=target_variable_name)

    metrics = {}

    for model in trained_models:

        ml_algorithm, iteration = model.split("_")

        if not iteration in metrics:
            metrics[iteration] = {}

        iteration_df = predictions.loc[predictions[iterator] == iteration]

        result_cols = [col for col in iteration_df.columns if ml_algorithm in col]
        predicted_col = result_cols.pop(0)

        metrics[iteration][ml_algorithm] = {}

        for set in iteration_df[Results.Set.value].unique():

            set_df = iteration_df.loc[iteration_df[Results.Set.value] == set]

            metrics[iteration][ml_algorithm][set] = calculate_metrics(
                y_true=set_df[label_col_name],
                y_pred=set_df[predicted_col],
                y_probs=(
                    set_df[result_cols] if problem_type == ProblemTypes.Classification else None
                ),
                problem_type=problem_type,
            )

    return metrics


def calculate_metrics(y_true, y_pred, y_probs, problem_type):
    """
    Calculates evaluation metrics based on the problem type.

    Args:
        y_true (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.
        problem_type (str): Type of the problem ('classification' or 'regression').

    Returns:
        dict: Dictionary containing calculated metrics.
    """
    if problem_type == ProblemTypes.Classification:

        return {
            EvaluationMetrics.Accuracy: accuracy_score(y_true, y_pred),
            EvaluationMetrics.Precision: precision_score(y_true, y_pred),
            EvaluationMetrics.Recall: recall_score(
                y_true, y_pred
            ),  # Also known as sensitivity or true positive rate
            EvaluationMetrics.Specificity: specificity_score(
                y_true, y_pred
            ),  # Recall for negative samples or True negative rate
            EvaluationMetrics.AUROC: (
                roc_auc_score(y_true=y_true, y_score=y_probs) if y_probs is not None else None
            ),
            EvaluationMetrics.MCC: mcc(y_true, y_pred),
            EvaluationMetrics.F1Score: f1_score(y_true, y_pred),
            EvaluationMetrics.GScore: gmean(y_true, y_pred),
        }
    else:

        return {
            EvaluationMetrics.RMSE: sqrt(mean_squared_error(y_true, y_pred)),
            EvaluationMetrics.R2: r2_score(y_true, y_pred),
            EvaluationMetrics.MAE: mean_absolute_error(y_true, y_pred),
            # "sep": calculate_standard_error(y_true, y_pred),
        }


def plot_results(
    predictions: pd.DataFrame,
    split_type: SplitTypes,
    target_variable_name: str,
    ml_algorithms: list,
    problem_type: ProblemTypes,
    save_path: Path,
    class_names: dict = None,
):
    iterator = get_iterator_name(split_type)

    label_col_name = get_true_label_column_name(target_variable_name=target_variable_name)

    for model in ml_algorithms:

        ml_algorithm, iteration = model.split("_")

        results_df = predictions.loc[
            (predictions[iterator] == iteration)
            & (predictions[Results.Set.value] == DataSets.Test.value)
        ]

        result_cols = [col for col in results_df.columns if ml_algorithm in col]
        predicted_col = result_cols.pop(0)

        if problem_type == ProblemTypes.Classification:

            fig = plot_confusion_matrix(
                y_true=results_df[label_col_name],
                y_pred=results_df[predicted_col],
                display_labels=(list(class_names.values()) if class_names else None),
                title=f"{target_variable_name}\nConfusion Matrix for\n {ml_algorithm} - {iteration}",
            )
            save_plot_path = save_path / f"{ml_algorithm}_{iteration}_confusion_matrix.png"
            save_plot(fig=fig, path=save_plot_path)

            for class_num, probs_col in enumerate(result_cols):

                if len(result_cols) == 1:
                    class_num = 1

                fig = plot_auroc(
                    y_true=results_df[label_col_name],
                    y_scores=results_df[probs_col],
                    title=f"{target_variable_name}\nROC Curve for\n {ml_algorithm} Class {class_num} - {iteration}",
                )
                save_plot_path = (
                    save_path / f"{ml_algorithm}_{iteration}_class_{class_num}_roc_curve.png"
                )
                save_plot(fig=fig, path=save_plot_path)

        elif problem_type == ProblemTypes.Regression:
            fig = plot_parity(
                y_true=results_df[label_col_name],
                y_pred=results_df[predicted_col],
                title=f"{target_variable_name}\nParity Plot for\n {ml_algorithm} - {iteration}",
            )
            save_plot_path = save_path / f"{ml_algorithm}_{iteration}_parity_plot.png"
            save_plot(fig=fig, path=save_plot_path)

    return None
