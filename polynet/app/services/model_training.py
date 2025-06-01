from math import sqrt

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
import streamlit as st
from torch import load, save

from polynet.options.enums import EvaluationMetrics, ProblemTypes


def split_data(data, test_size=0.2, random_state=1, stratify=None):
    """
    Splits the data into training and testing sets.

    Args:
        data (pd.DataFrame): The input data.
        target_col (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
        X_train, X_test, y_train, y_test: The training and testing sets.
    """

    return train_test_split(data, test_size=test_size, random_state=random_state, stratify=stratify)


def save_gnn_model(model, path):
    """
    Saves the GNN model to the specified path.

    Args:
        model: The GNN model to save.
        path (str): The path where the model will be saved.
    """

    save(model, path)


def load_gnn_model(path):
    """
    Loads the GNN model from the specified path.

    Args:
        path (str): The path from which to load the model.

    Returns:
        The loaded GNN model.
    """
    return load(path, weights_only=False)


def save_plot(fig, path, dpi=300):
    """
    Saves the plot to the specified path.

    Args:
        fig: The figure to save.
        path (str): The path where the plot will be saved.
    """
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    print(f"Plot saved to {path}")


def calculate_standard_error(actual_values, predicted_values):
    return np.sqrt(np.mean((actual_values - predicted_values) ** 2))


def calculate_sep(y_true, y_pred):
    """
    Calculates the Standard Error of Prediction (SEP).

    Parameters:
    - y_true: array-like of shape (n_samples,), ground truth values.
    - y_pred: array-like of shape (n_samples,), predicted values.

    Returns:
    - sep: float, the standard error of prediction.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")

    residuals = y_true - y_pred
    sep = np.sqrt(np.sum(residuals**2) / (len(residuals) - 1))
    return sep


def calculate_metrics(y_true, y_pred, problem_type):
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
            EvaluationMetrics.F1Score: f1_score(y_true, y_pred, average="weighted"),
            EvaluationMetrics.Precision: precision_score(y_true, y_pred, average="weighted"),
            EvaluationMetrics.Recall: recall_score(y_true, y_pred, average="weighted"),
        }
    else:

        return {
            EvaluationMetrics.RMSE: sqrt(mean_squared_error(y_true, y_pred)),
            EvaluationMetrics.R2: r2_score(y_true, y_pred),
            EvaluationMetrics.MAE: mean_absolute_error(y_true, y_pred),
            # "sep": calculate_standard_error(y_true, y_pred),
        }
