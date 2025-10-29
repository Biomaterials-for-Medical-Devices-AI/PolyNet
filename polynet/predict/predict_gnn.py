import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader

from polynet.options.col_names import (
    get_iterator_name,
    get_predicted_label_column_name,
    get_true_label_column_name,
)
from polynet.options.enums import DataSets, ProblemTypes, Results, SplitTypes
from polynet.utils import prepare_probs_df


def get_predictions_df_gnn(
    models: dict,
    loaders: dict,
    problem_type: ProblemTypes,
    split_type: SplitTypes,
    target_variable_name: str | None = None,
):

    label_col_name = get_true_label_column_name(target_variable_name=target_variable_name)
    iterator = get_iterator_name(split_type)

    all_dfs = []
    predictions_all = pd.DataFrame()
    last_iteration = None

    for model_name, model in models.items():
        gnn_arch, iteration = model_name.split("_")

        predicted_col_name = get_predicted_label_column_name(
            target_variable_name=target_variable_name, model_name=gnn_arch
        )

        train_loader, val_loader, test_loader = loaders[iteration]
        train_loader = DataLoader(train_loader.dataset, batch_size=1, shuffle=False)
        val_loader = DataLoader(val_loader.dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_loader.dataset, batch_size=1, shuffle=False)

        train_preds = model.predict_loader(train_loader)

        train_df = pd.DataFrame(
            {
                Results.Index.value: train_preds[0],
                Results.Set.value: DataSets.Training.value,
                label_col_name: np.concatenate(
                    [mol.y.cpu().detach().numpy() for mol in train_loader]
                ),
                predicted_col_name: train_preds[1],
            }
        )

        val_preds = model.predict_loader(val_loader)

        val_df = pd.DataFrame(
            {
                Results.Index.value: val_preds[0],
                Results.Set.value: DataSets.Validation.value,
                label_col_name: np.concatenate(
                    [mol.y.cpu().detach().numpy() for mol in val_loader]
                ),
                predicted_col_name: val_preds[1],
            }
        )

        test_preds = model.predict_loader(test_loader)

        test_df = pd.DataFrame(
            {
                Results.Index.value: test_preds[0],
                Results.Set.value: DataSets.Test.value,
                label_col_name: np.concatenate(
                    [mol.y.cpu().detach().numpy() for mol in test_loader]
                ),
                predicted_col_name: test_preds[1],
            }
        )

        if problem_type == ProblemTypes.Classification:
            probs_train = prepare_probs_df(
                probs=train_preds[-1],
                target_variable_name=target_variable_name,
                model_name=gnn_arch,
            )
            train_df[probs_train.columns] = probs_train.to_numpy()

            probs_val = prepare_probs_df(
                probs=val_preds[-1], target_variable_name=target_variable_name, model_name=gnn_arch
            )
            val_df[probs_val.columns] = probs_val.to_numpy()

            probs_test = prepare_probs_df(
                probs=test_preds[-1], target_variable_name=target_variable_name, model_name=gnn_arch
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
