from math import sqrt

from imblearn.metrics import geometric_mean_score as gmean
from imblearn.metrics import specificity_score
import joblib
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
import torch
from torch import load, save
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from polynet.app.options.file_paths import gnn_model_dir
from polynet.options.enums import EvaluationMetrics, ProblemTypes


def save_tml_model(model, path):
    joblib.dump(model, path)


def load_tml_model(path):
    return joblib.load(path)


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


def load_tml_model(path):

    return joblib.load(path)


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
        termination = str(model_file).split(".")[-1]
        if termination == "pt":
            models[model_name] = load_gnn_model(model_file)
        else:
            models[model_name] = load_tml_model(model_file)

    return models


def load_scalers_from_experiment(experiment_path: str, model_names: list) -> dict:
    """
    Loads trained GNN models from the specified experiment path.

    Args:
        experiment_path (str): Path to the experiment directory.

    Returns:
        dict: Dictionary containing model names as keys and loaded models as values.
    """

    gnn_models_path = gnn_model_dir(experiment_path)
    scaler = {}

    for model_name in model_names:

        model_name = model_name.split(".")[0]
        scaler_name = model_name.split("-")[-1]

        scaler_file = gnn_models_path / f"{scaler_name}.pkl"

        scaler[scaler_name] = load_tml_model(scaler_file)

    return scaler


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
