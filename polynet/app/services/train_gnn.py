from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from polynet.app.options.data import DataOptions
from polynet.app.options.file_paths import (
    gnn_raw_data_file,
    gnn_raw_data_path,
    polynet_experiments_base_dir,
)
from polynet.app.options.general_experiment import GeneralConfigOptions
from polynet.app.options.representation import RepresentationOptions
from polynet.app.options.train_GNN import TrainGNNOptions
from polynet.options.col_names import (
    get_predicted_label_column_name,
    get_score_column_name,
    get_true_label_column_name,
)
from polynet.call_methods import (
    compute_class_weights,
    create_network,
    make_loss,
    make_optimizer,
    make_scheduler,
)
from polynet.featurizer.graph_representation.polymer import CustomPolymerGraph
from polynet.options.enums import (
    DataSets,
    NetworkParams,
    Optimizers,
    ProblemTypes,
    Results,
    Schedulers,
)
from polynet.utils.model_training import gnn_hyp_opt, predict_network, train_model


def predict_gnn_model(model, loaders, target_variable_name=None):

    train_loader, val_loader, test_loader = loaders

    idx, y_true, y_pred, y_score = predict_network(model, train_loader)
    predictions_train = create_results_dataframe(
        target_variable_name=target_variable_name,
        idx=idx,
        y_pred=y_pred,
        y_true=y_true,
        y_score=y_score,
        set_name=DataSets.Training.value,
        model_name=model._name,
    )

    idx, y_true, y_pred, y_score = predict_network(model, val_loader)
    predictions_val = create_results_dataframe(
        target_variable_name=target_variable_name,
        idx=idx,
        y_pred=y_pred,
        y_true=y_true,
        y_score=y_score,
        set_name=DataSets.Validation.value,
        model_name=model._name,
    )
    idx, y_true, y_pred, y_score = predict_network(model, test_loader)
    prediction_test = create_results_dataframe(
        target_variable_name=target_variable_name,
        idx=idx,
        y_pred=y_pred,
        y_true=y_true,
        y_score=y_score,
        set_name=DataSets.Test.value,
        model_name=model._name,
    )

    predictions = pd.concat(
        [predictions_train, predictions_val, prediction_test], ignore_index=True
    )

    return predictions


def create_results_dataframe(
    target_variable_name: str,
    idx: list,
    y_pred: list,
    y_true: list,
    y_score: list,
    set_name: str,
    model_name: str = None,
):

    true_label = get_true_label_column_name(target_variable_name=target_variable_name)
    predicted_label = get_predicted_label_column_name(
        target_variable_name=target_variable_name, model_name=model_name
    )

    results = pd.DataFrame(
        {
            Results.Index.value: idx,
            Results.Set.value: set_name,
            true_label: y_true,
            predicted_label: y_pred,
        }
    )

    if y_score is not None:
        probs = prepare_probs_df(
            probs=y_score, target_variable_name=target_variable_name, model_name=model_name
        )
        results = pd.concat([results, probs], axis=1)

    return results


def prepare_probs_df(probs: np.ndarray, target_variable_name: str = None, model_name: str = None):
    """
    Convert probability predictions into a DataFrame.

    - For binary classification (2 classes), include only the second class (index 1).
    - For multi-class classification (3+ classes), include a column per class.

    Args:
        probs (np.ndarray): Array of shape (n_samples, n_classes)
        target_variable_name (str): Name of the target variable
        model_name (str): Name of the model

    Returns:
        pd.DataFrame: A DataFrame with appropriately named probability columns
    """
    n_classes = probs.shape[1] if probs.ndim > 1 else 1
    probs_df = pd.DataFrame()

    if n_classes == 2:
        col_name = get_score_column_name(
            target_variable_name=target_variable_name, model_name=model_name
        )
        # Binary classification: only use the second class (probability of class 1)
        probs_df[f"{col_name}"] = probs[:, 1]
    else:
        # Multi-class classification: create one column per class
        for i in range(n_classes):
            col_name = get_score_column_name(
                target_variable_name=target_variable_name, model_name=model_name, class_num=i
            )
            probs_df[f"{col_name} {i}"] = probs[:, i]

    return probs_df
