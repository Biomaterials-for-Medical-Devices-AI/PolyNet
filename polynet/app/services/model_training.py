from math import sqrt

from imblearn.metrics import geometric_mean_score as gmean
from imblearn.metrics import specificity_score
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.model_selection import train_test_split
import streamlit as st
import torch
from torch import load, save
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from polynet.app.options.data import DataOptions
from polynet.app.options.file_paths import gnn_model_dir
from polynet.app.options.general_experiment import GeneralConfigOptions
from polynet.options.enums import EvaluationMetrics, ProblemTypes, SplitMethods, SplitTypes
from polynet.utils.data_preprocessing import class_balancer


def split_data(data, test_size, random_state, stratify=None):
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


def get_data_split_indices(
    data: pd.DataFrame, data_options: DataOptions, general_experiment_options: GeneralConfigOptions
):

    if general_experiment_options.split_type == SplitTypes.TrainValTest:

        train_data_idxs, val_data_idxs, test_data_idxs = [], [], []

        for i in range(general_experiment_options.n_bootstrap_iterations):
            # Initial train-test split
            train_data, test_data = split_data(
                data=data,
                test_size=general_experiment_options.test_ratio,
                stratify=(
                    data[data_options.target_variable_col]
                    if general_experiment_options.split_method == SplitMethods.Stratified
                    else None
                ),
                random_state=general_experiment_options.random_seed + i,
            )

            # Optional class balancing on training set
            if general_experiment_options.train_set_balance:
                train_data = class_balancer(
                    data=train_data,
                    target=data_options.target_variable_col,
                    desired_class_proportion=general_experiment_options.train_set_balance,
                    random_state=general_experiment_options.random_seed + i,
                )

            # Further split train into train/validation
            train_data, val_data = split_data(
                data=train_data,
                test_size=general_experiment_options.val_ratio,
                stratify=(
                    train_data[data_options.target_variable_col]
                    if general_experiment_options.split_method == SplitMethods.Stratified
                    else None
                ),
                random_state=general_experiment_options.random_seed + i,
            )

            train_data_idxs.append(train_data.index.tolist())
            val_data_idxs.append(val_data.index.tolist())
            test_data_idxs.append(test_data.index.tolist())

    return train_data_idxs, val_data_idxs, test_data_idxs


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
            EvaluationMetrics.AUROC: roc_auc_score(y_true=y_true, y_score=y_probs),
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


def load_models_from_experiment(experiment_path: str, model_names: list) -> dict:
    """
    Loads trained GNN models from the specified experiment path.

    Args:
        experiment_path (str): Path to the experiment directory.

    Returns:
        dict: Dictionary containing model names as keys and loaded models as values.
    """

    gnn_models_path = gnn_model_dir(experiment_path)
    models = {}

    for model_name in model_names:
        model_file = gnn_models_path / model_name
        model_name = model_file.stem
        models[model_name] = load_gnn_model(model_file)

    return models


def predict_network(models: dict, dataset: Dataset) -> pd.DataFrame:

    indexes = [data.idx for data in dataset]
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    predictions_models = {}

    for model_name, model in models.items():
        model.eval()
        predictions = []

        with torch.no_grad():
            for batch in loader:

                preds = model.predict(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch_index=batch.batch,
                    monomer_weight=batch.weight_monomer,
                )

                predictions.append(preds)

        predictions_models[model_name] = pd.Series(
            np.concatenate(predictions), index=indexes, name=model_name
        )

    predictions_models = pd.DataFrame(predictions_models)

    return predictions_models
