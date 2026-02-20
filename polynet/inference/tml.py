"""
polynet.inference.tml
======================
Assembles standardised predictions DataFrames from trained TML models.

Takes the ``trained_models`` and ``training_data`` dicts returned by
``train_tml_ensemble`` and produces a single wide DataFrame with one
row per sample per iteration, consistent with the GNN predictions format.

Note: TML inference produces only train and test splits. The validation
set was merged into training during ``train_tml_ensemble``, so there is
no separate validation split to report.

Public API
----------
::

    from polynet.inference.tml import get_predictions_df_tml
"""

from __future__ import annotations

import logging

import pandas as pd

from polynet.config.column_names import (
    get_iterator_name,
    get_predicted_label_column_name,
    get_true_label_column_name,
)
from polynet.config.constants import DataSet, ResultColumn
from polynet.config.enums import ProblemType, SplitType
from polynet.inference.utils import assemble_predictions, prepare_probs_df

logger = logging.getLogger(__name__)


def get_predictions_df_tml(
    models: dict,
    training_data: dict,
    split_type: SplitType | str,
    target_variable_col: str,
    problem_type: ProblemType | str,
    target_variable_name: str | None = None,
) -> pd.DataFrame:
    """
    Collect TML predictions across all models and iterations into a DataFrame.

    For each trained model, runs inference on the train and test DataFrames
    and assembles a wide DataFrame where multiple models from the same
    iteration appear as separate columns.

    Note: There is no validation split in TML inference. Validation indices
    were merged into the training set during ``train_tml_ensemble`` to
    maximise the data available for HPO cross-validation.

    Parameters
    ----------
    models:
        Dict of ``{"{algo}-{df_name}_{iteration}": fitted_model}`` as
        returned by ``train_tml_ensemble``.
    training_data:
        Dict of ``{"{df_name}_{iteration}": (train_df, test_df)}``
        as returned by ``train_tml_ensemble``.
    split_type:
        The split strategy used — determines the iterator column name.
    target_variable_col:
        Column name of the target variable in the DataFrames.
    problem_type:
        Classification or regression — determines whether probability
        columns are added.
    target_variable_name:
        Human-readable name of the target property for column naming.
        Defaults to ``target_variable_col`` if not provided.

    Returns
    -------
    pd.DataFrame
        Wide predictions DataFrame with columns:
        - Sample index, set label (train/test), iterator, true labels
        - One predicted column per model
        - One probability column per class per model (classification only)
    """
    problem_type = ProblemType(problem_type) if isinstance(problem_type, str) else problem_type
    split_type = SplitType(split_type) if isinstance(split_type, str) else split_type

    display_name = target_variable_name or target_variable_col
    label_col = get_true_label_column_name(display_name)
    iterator = get_iterator_name(split_type)

    per_model_dfs: list[tuple[str, pd.DataFrame]] = []

    for model_name, model in models.items():
        # model_name format: "{algo}-{df_name}_{iteration}"
        ml_model, iteration = model_name.split("_", 1)
        ml_algorithm, df_name = ml_model.split("-", 1)

        predicted_col = get_predicted_label_column_name(display_name, ml_model)

        train_df, test_df = training_data[f"{df_name}_{iteration}"]

        splits = [(train_df, DataSet.Training), (test_df, DataSet.Test)]

        split_dfs: list[pd.DataFrame] = []

        for df, set_label in splits:
            X = df.iloc[:, :-1]
            y_true = df[target_variable_col]
            y_pred = model.predict(X)

            split_df = pd.DataFrame(
                {
                    ResultColumn.Index: df.index,
                    ResultColumn.Set: set_label,
                    label_col: y_true.values,
                    predicted_col: y_pred,
                }
            )

            if problem_type == ProblemType.Classification:
                y_score = model.predict_proba(X)
                probs_df = prepare_probs_df(
                    probs=y_score, target_variable_name=display_name, model_name=ml_model
                )
                split_df[probs_df.columns] = probs_df.to_numpy()

            split_dfs.append(split_df)

        predictions_df = pd.concat(split_dfs, ignore_index=True)
        predictions_df[iterator] = iteration
        predictions_df = predictions_df[~predictions_df[ResultColumn.Index].duplicated(keep="last")]

        per_model_dfs.append((iteration, predictions_df))

    predictions = assemble_predictions(per_model_dfs, iterator, ResultColumn.Index)

    meta_cols = [ResultColumn.Index, ResultColumn.Set, iterator, label_col]
    pred_cols = [col for col in predictions.columns if col not in meta_cols]

    return predictions[meta_cols + pred_cols]
