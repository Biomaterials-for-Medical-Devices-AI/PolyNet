"""
polynet.config.column_names
============================
Helper functions for generating standardised result DataFrame column names.

These functions build column names dynamically based on the target variable
name and model name, ensuring consistency across the entire pipeline —
from training results to plots to exports.

All returned strings reference the constants in ``polynet.config.constants``
so that column names have a single source of truth.
"""

from polynet.config.constants import ResultColumn
from polynet.config.enums import IteratorType, SplitType


def get_iterator_name(split_type: SplitType) -> str:
    """
    Return the appropriate iterator label for a given split type.

    Used to label rows in results DataFrames produced during training loops.

    Parameters
    ----------
    split_type:
        The data splitting strategy in use.

    Returns
    -------
    str
        A human-readable iterator label (value of the relevant IteratorType).
    """
    match split_type:
        case SplitType.TrainValTest:
            return IteratorType.BootstrapIteration.value
        case SplitType.CrossValidation:
            return IteratorType.Fold.value
        case _:
            return IteratorType.Iteration.value


def get_true_label_column_name(target_variable_name: str | None) -> str:
    """
    Return the column name for ground-truth target values.

    Parameters
    ----------
    target_variable_name:
        Human-readable name of the target variable. If None or empty,
        returns the bare label constant.

    Returns
    -------
    str
        Column name string, e.g. ``"True Tg"`` or ``"True"``.
    """
    if target_variable_name:
        return f"{ResultColumn.LABEL} {target_variable_name}"
    return ResultColumn.LABEL


def get_predicted_label_column_name(
    target_variable_name: str | None, model_name: str | None = None
) -> str:
    """
    Return the column name for model-predicted values.

    Parameters
    ----------
    target_variable_name:
        Human-readable name of the target variable.
    model_name:
        Name of the model producing the predictions. Optional.

    Returns
    -------
    str
        Column name string, e.g. ``"GCN Predicted Tg"`` or ``"Predicted"``.
    """
    if model_name and target_variable_name:
        return f"{model_name} {ResultColumn.PREDICTED} {target_variable_name}"
    if target_variable_name:
        return f"{ResultColumn.PREDICTED} {target_variable_name}"
    if model_name:
        return f"{model_name} {ResultColumn.PREDICTED}"
    return ResultColumn.PREDICTED


def get_score_column_name(
    target_variable_name: str | None, model_name: str | None = None, class_num: int = 1
) -> str:
    """
    Return the column name for predicted class probability scores.

    Used for classification tasks where soft probabilities are stored
    alongside hard predictions.

    Parameters
    ----------
    target_variable_name:
        Human-readable name of the target variable.
    model_name:
        Name of the model producing the scores. Optional.
    class_num:
        The class index this score column refers to. Defaults to 1
        (positive class in binary classification).

    Returns
    -------
    str
        Column name string, e.g. ``"GCN Score Tg 1"`` or ``"Score 1"``.
    """
    if model_name and target_variable_name:
        return f"{model_name} {ResultColumn.SCORE} {target_variable_name} {class_num}"
    if target_variable_name:
        return f"{ResultColumn.SCORE} {target_variable_name} {class_num}"
    if model_name:
        return f"{model_name} {ResultColumn.SCORE} {class_num}"
    return f"{ResultColumn.SCORE} {class_num}"
