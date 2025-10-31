import pandas as pd

from polynet.options.col_names import (
    get_iterator_name,
    get_predicted_label_column_name,
    get_true_label_column_name,
)
from polynet.options.enums import DataSets, ProblemTypes, Results, SplitTypes
from polynet.utils import prepare_probs_df


def get_predictions_df_tml(
    models: dict,
    dataframes: dict,
    split_type: SplitTypes,
    target_variable_col: str,
    problem_type: ProblemTypes,
    target_variable_name: str = None,
) -> pd.DataFrame:

    label_col_name = get_true_label_column_name(target_variable_name=target_variable_name)
    iterator = get_iterator_name(split_type)

    all_dfs = []
    predictions_all = pd.DataFrame()
    last_iteration = None

    for model_name, model in models.items():

        ml_model, iteration = model_name.split("_")
        ml_algorithm, df_name = ml_model.split("-")

        predicted_col_name = get_predicted_label_column_name(
            target_variable_name=target_variable_name, model_name=ml_model
        )

        train_data, val_data, test_data = dataframes[f"{df_name}_{iteration}"]

        train_preds = model.predict(train_data.iloc[:, :-1])
        train_df = pd.DataFrame(
            {
                Results.Index.value: train_data.index,
                Results.Set.value: DataSets.Training.value,
                label_col_name: train_data[target_variable_col],
                predicted_col_name: train_preds,
            }
        )

        val_preds = model.predict(val_data.iloc[:, :-1])
        val_df = pd.DataFrame(
            {
                Results.Index.value: val_data.index,
                Results.Set.value: DataSets.Validation.value,
                label_col_name: val_data[target_variable_col],
                predicted_col_name: val_preds,
            }
        )

        test_preds = model.predict(test_data.iloc[:, :-1])
        test_df = pd.DataFrame(
            {
                Results.Index.value: test_data.index,
                Results.Set.value: DataSets.Test.value,
                label_col_name: test_data[target_variable_col],
                predicted_col_name: test_preds,
            }
        )

        if problem_type == ProblemTypes.Classification:

            probs_train = model.predict_proba(train_data.iloc[:, :-1])
            probs_train = prepare_probs_df(
                probs=probs_train, target_variable_name=target_variable_name, model_name=ml_model
            )

            train_df[probs_train.columns] = probs_train.to_numpy()

            probs_val = model.predict_proba(val_data.iloc[:, :-1])
            probs_val = prepare_probs_df(
                probs=probs_val, target_variable_name=target_variable_name, model_name=ml_model
            )
            val_df[probs_val.columns] = probs_val.to_numpy()

            probs_test = model.predict_proba(test_data.iloc[:, :-1])
            probs_test = prepare_probs_df(
                probs=probs_test, target_variable_name=target_variable_name, model_name=ml_model
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
