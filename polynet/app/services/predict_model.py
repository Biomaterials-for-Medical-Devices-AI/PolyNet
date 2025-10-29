from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch.nn import Module
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from polynet.app.options.data import DataOptions
from polynet.app.options.representation import RepresentationOptions
from polynet.app.options.train_TML import TrainTMLOptions
from polynet.app.services.model_training import calculate_metrics, save_plot
from polynet.app.services.train_gnn import prepare_probs_df
from polynet.app.services.train_tml import load_dataframes, transform_dependent_variables
from polynet.options.col_names import (
    get_iterator_name,
    get_predicted_label_column_name,
    get_true_label_column_name,
)
from polynet.options.enums import DataSets, ProblemTypes, Results, SplitTypes, TransformDescriptors
from polynet.utils.plot_utils import (
    plot_auroc,
    plot_confusion_matrix,
    plot_learning_curve,
    plot_parity,
)


def get_predictions_df_tml(
    models: dict, dataframes: dict, split_type: SplitTypes, data_options: DataOptions
) -> pd.DataFrame:

    label_col_name = get_true_label_column_name(
        target_variable_name=data_options.target_variable_name
    )
    iterator = get_iterator_name(split_type)

    all_dfs = []
    predictions_all = pd.DataFrame()
    last_iteration = None

    for model_name, model in models.items():

        ml_model, iteration = model_name.split("_")
        ml_algorithm, df_name = ml_model.split("-")

        predicted_col_name = get_predicted_label_column_name(
            target_variable_name=data_options.target_variable_name, model_name=ml_model
        )

        train_data, val_data, test_data = dataframes[f"{df_name}_{iteration}"]

        train_preds = model.predict(train_data.iloc[:, :-1])
        train_df = pd.DataFrame(
            {
                Results.Index.value: train_data.index,
                Results.Set.value: DataSets.Training.value,
                label_col_name: train_data[data_options.target_variable_col],
                predicted_col_name: train_preds,
            }
        )

        val_preds = model.predict(val_data.iloc[:, :-1])
        val_df = pd.DataFrame(
            {
                Results.Index.value: val_data.index,
                Results.Set.value: DataSets.Validation.value,
                label_col_name: val_data[data_options.target_variable_col],
                predicted_col_name: val_preds,
            }
        )

        test_preds = model.predict(test_data.iloc[:, :-1])
        test_df = pd.DataFrame(
            {
                Results.Index.value: test_data.index,
                Results.Set.value: DataSets.Test.value,
                label_col_name: test_data[data_options.target_variable_col],
                predicted_col_name: test_preds,
            }
        )

        if data_options.problem_type == ProblemTypes.Classification:

            probs_train = model.predict_proba(train_data.iloc[:, :-1])
            probs_train = prepare_probs_df(
                probs=probs_train,
                target_variable_name=data_options.target_variable_name,
                model_name=ml_model,
            )

            train_df[probs_train.columns] = probs_train.to_numpy()

            probs_val = model.predict_proba(val_data.iloc[:, :-1])
            probs_val = prepare_probs_df(
                probs=probs_val,
                target_variable_name=data_options.target_variable_name,
                model_name=ml_model,
            )
            val_df[probs_val.columns] = probs_val.to_numpy()

            probs_test = model.predict_proba(test_data.iloc[:, :-1])
            probs_test = prepare_probs_df(
                probs=probs_test,
                target_variable_name=data_options.target_variable_name,
                model_name=ml_model,
            )

            test_df[probs_test.columns] = probs_test.to_numpy()

        predictions_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        predictions_df[iterator] = iteration
        predictions_df = predictions_df[
            ~predictions_df[Results.Index.value].duplicated(keep="last")
        ]

        if last_iteration is None:
            predictions_all = predictions_df.copy()
            last_iteration = iteration
        elif iteration == last_iteration:
            new_cols = [col for col in predictions_df.columns if col not in predictions_all.columns]
            predictions_df = predictions_df[new_cols]
            predictions_all = pd.concat([predictions_all, predictions_df], axis=1)
        else:
            all_dfs.append(predictions_all.copy())
            predictions_all = predictions_df.copy()
            last_iteration = iteration

    all_dfs.append(predictions_all)
    predictions = pd.concat(all_dfs)

    cols = [Results.Index.value, Results.Set.value, iterator, label_col_name]
    cols += [col for col in predictions if col not in cols]

    return predictions[cols]


def plot_learning_curves(models: dict, save_path: Path):
    for model_name, model in models.items():

        losses = model.losses
        title = f"{model_name} Learning Curve"
        learning_curve = plot_learning_curve(losses, title=title)

        save_plot_path = save_path / f"{model_name}_learning_curve.png"
        save_plot(fig=learning_curve, path=save_plot_path)

    return


def predict_tml_model(
    model: dict,
    fit_data: pd.DataFrame,
    transform_data: pd.DataFrame,
    train_tml_options: TrainTMLOptions,
    predicted_col_name="Predictions",
):

    if train_tml_options.TransformFeatures != TransformDescriptors.NoTransformation:

        original_train_features = fit_data.iloc[:, :-1].copy()

        train_features = transform_dependent_variables(
            fit_data=original_train_features,
            transform_data=fit_data.iloc[:, :-1],
            transform_type=train_tml_options.TransformFeatures,
        )
        fit_data.iloc[:, :-1] = train_features

        val_features = transform_dependent_variables(
            fit_data=original_train_features,
            transform_data=transform_data.iloc[:, :-1],
            transform_type=train_tml_options.TransformFeatures,
        )
        transform_data.iloc[:, :-1] = val_features

    predictions = model.predict(transform_data.iloc[:, :-1])

    predictions_df = pd.DataFrame(
        predictions, index=transform_data.index, columns=predicted_col_name
    )

    return predictions_df


def predict_unseen_tml(models: dict, scalers: dict, dfs: dict, data_options: DataOptions):

    label_col_name = get_true_label_column_name(
        target_variable_name=data_options.target_variable_name
    )
    predictions_all = None

    for model_name, model in models.items():

        ml_model, iteration = model_name.split("_")
        ml_algorithm, df_name = ml_model.split("-")

        model_log_name = model_name.replace("_", " ")

        predicted_col_name = get_predicted_label_column_name(
            target_variable_name=data_options.target_variable_name, model_name=model_log_name
        )

        df = dfs[df_name]

        if scalers:
            scaler_name = model_name.split("-")[-1]
            scaler = scalers[scaler_name]
            df_cols = df.columns
            df = scaler.transform(df)
            df = pd.DataFrame(df, columns=df_cols)

        preds = model.predict(df)

        preds_df = pd.DataFrame({predicted_col_name: preds})

        if data_options.problem_type == ProblemTypes.Classification:
            probs_df = prepare_probs_df(
                probs=model.predict_proba(df),
                target_variable_name=data_options.target_variable_name,
                model_name=model_log_name,
            )
            preds_df[probs_df.columns] = probs_df.to_numpy()

        if predictions_all is None:
            predictions_all = preds_df.copy()
        else:
            predictions_all = pd.concat([predictions_all, preds_df], axis=1)

    return predictions_all


def predict_unseen_gnn(models: dict, dataset: Dataset, data_options: DataOptions):

    predictions_all = None

    for model_name, model in models.items():

        model_name = model_name.replace("_", " ")

        predicted_col_name = get_predicted_label_column_name(
            target_variable_name=data_options.target_variable_name, model_name=model_name
        )

        loader = DataLoader(dataset)

        preds = model.predict_loader(loader)

        preds_df = pd.DataFrame({Results.Index.value: preds[0], predicted_col_name: preds[1]})

        if data_options.problem_type == ProblemTypes.Classification:
            probs_df = prepare_probs_df(
                probs=preds[-1],
                target_variable_name=data_options.target_variable_name,
                model_name=model_name,
            )
            preds_df[probs_df.columns] = probs_df.to_numpy()

        if predictions_all is None:
            predictions_all = preds_df.copy()
        else:
            predictions_all = pd.merge(left=predictions_all, right=preds_df, on=[Results.Index])

    return predictions_all


def get_gnn_model_predictions(model: Module, loader: DataLoader, prediction_col_name="Predictions"):
    predictions = []
    idxs = []

    for batch in loader:
        model.eval()
        with torch.no_grad():
            output = model.predict(batch.x, batch.edge_index, batch.batch)
            predictions.append(output)
            idxs.append(batch.idx)

    predictions = np.concatenate(predictions, axis=0)
    idxs = np.concatenate(idxs, axis=0)

    predictions = pd.DataFrame(predictions, index=idxs, columns=prediction_col_name)

    return predictions


def get_gnn_model_probs(model: Module, loader: DataLoader, probs_col_name="Probabilities"):
    predictions = []
    idxs = []

    for batch in loader:
        model.eval()
        with torch.no_grad():
            output = model.predict(batch.x, batch.edge_index, batch.batch)
            predictions.append(output)
            idxs.append(batch.idx)

    predictions = np.concatenate(predictions, axis=0)
    idxs = np.concatenate(idxs, axis=0)

    return predictions
