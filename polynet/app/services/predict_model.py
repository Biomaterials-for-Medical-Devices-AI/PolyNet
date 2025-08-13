import streamlit as st

from pathlib import Path
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
import torch
from torch.nn import Module
from polynet.app.services.train_tml import load_dataframes, transform_dependent_variables
from polynet.utils.plot_utils import plot_auroc, plot_confusion_matrix, plot_parity
from polynet.app.options.representation import RepresentationOptions
from polynet.app.options.data import DataOptions
from polynet.app.options.train_TML import TrainTMLOptions
from polynet.options.enums import (
    ProblemTypes,
    SplitTypes,
    TransformDescriptors,
    Results,
    DataSets,
    SplitTypes,
)

from polynet.app.utils import (
    get_true_label_column_name,
    get_iterator_name,
    get_predicted_label_column_name,
)
from polynet.app.services.train_gnn import prepare_probs_df

from polynet.app.services.model_training import calculate_metrics, save_plot


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

        iteration, ml_model = model_name.split("_")
        strs = ml_model.split(" ")
        ml_algorithm, df_name = strs[:-1], strs[-1]

        predicted_col_name = get_predicted_label_column_name(
            target_variable_name=data_options.target_variable_name, model_name=ml_model
        )

        train_data, val_data, test_data = dataframes[f"{iteration}_{df_name}"]

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

        iteration, ml_algorithm = model.split("_")

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


def plot_results(
    predictions: pd.DataFrame,
    split_type: SplitTypes,
    target_variable_name: str,
    ml_algorithms: list,
    problem_type: ProblemTypes,
    data_options: DataOptions,
    save_path: Path,
):
    iterator = get_iterator_name(split_type)

    label_col_name = get_true_label_column_name(target_variable_name=target_variable_name)

    for model in ml_algorithms:

        iteration, ml_algorithm = model.split("_")

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
                display_labels=(
                    list(data_options.class_names.values()) if data_options.class_names else None
                ),
                title=f"{data_options.target_variable_name}\nConfusion Matrix for\n {ml_algorithm} - {iteration}",
            )
            save_plot_path = save_path / f"{ml_algorithm}_{iteration}_confusion_matrix.png"
            save_plot(fig=fig, path=save_plot_path)

            for class_num, probs_col in enumerate(result_cols):

                if len(result_cols) == 1:
                    class_num = 1

                fig = plot_auroc(
                    y_true=results_df[label_col_name],
                    y_scores=results_df[probs_col],
                    title=f"{data_options.target_variable_name}\nROC Curve for\n {ml_algorithm} Class {class_num} - {iteration}",
                )
                save_plot_path = (
                    save_path / f"{ml_algorithm}_{iteration}_class_{class_num}_roc_curve.png"
                )
                save_plot(fig=fig, path=save_plot_path)

        elif problem_type == ProblemTypes.Regression:
            fig = plot_parity(
                y_true=results_df[label_col_name],
                y_pred=results_df[predicted_col],
                title=f"{data_options.target_variable_name}\nParity Plot for\n {ml_algorithm} - {iteration}",
            )
            save_plot_path = save_path / f"{ml_algorithm}_{iteration}_parity_plot.png"
            save_plot(fig=fig, path=save_plot_path)

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


def get_predictions_df_gnn(models: dict, loaders: dict, data_options, split_type):

    label_col_name = get_true_label_column_name(
        target_variable_name=data_options.target_variable_name
    )
    iterator = get_iterator_name(split_type)

    all_dfs = []
    predictions_all = pd.DataFrame()
    last_iteration = None

    for model_name, model in models.items():
        iteration, gnn_arch = model_name.split("_")

        predicted_col_name = get_predicted_label_column_name(
            target_variable_name=data_options.target_variable_name, model_name=gnn_arch
        )

        train_loader, val_loader, test_loader = loaders[iteration]
        train_loader = DataLoader(train_loader.dataset, batch_size=1, shuffle=False)

        idx, preds, y_vals, train_scores = model.predict_loader(train_loader)

        train_df = pd.DataFrame(
            {
                Results.Index.value: idx,
                Results.Set.value: DataSets.Training.value,
                label_col_name: y_vals,
                predicted_col_name: preds,
            }
        )

        idx, preds, y_vals, val_scores = model.predict_loader(val_loader)

        val_df = pd.DataFrame(
            {
                Results.Index.value: idx,
                Results.Set.value: DataSets.Validation.value,
                label_col_name: y_vals,
                predicted_col_name: preds,
            }
        )

        idx, preds, y_vals, test_scores = model.predict_loader(test_loader)

        test_df = pd.DataFrame(
            {
                Results.Index.value: idx,
                Results.Set.value: DataSets.Test.value,
                label_col_name: y_vals,
                predicted_col_name: preds,
            }
        )

        if data_options.problem_type == ProblemTypes.Classification:
            probs_train = prepare_probs_df(
                probs=train_scores,
                target_variable_name=data_options.target_variable_name,
                model_name=gnn_arch,
            )
            train_df[probs_train.columns] = probs_train.to_numpy()

            probs_val = prepare_probs_df(
                probs=val_scores,
                target_variable_name=data_options.target_variable_name,
                model_name=gnn_arch,
            )
            val_df[probs_val.columns] = probs_val.to_numpy()

            probs_test = prepare_probs_df(
                probs=test_scores,
                target_variable_name=data_options.target_variable_name,
                model_name=gnn_arch,
            )
            test_df[probs_test.columns] = probs_test.to_numpy()

        predictions_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        predictions_df[iterator] = iteration

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
