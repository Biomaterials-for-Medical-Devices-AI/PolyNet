"""
polynet.inference.gnn
======================
Assembles standardised predictions DataFrames from trained GNN models.

Takes the ``trained_models`` and ``loaders`` dicts returned by
``train_gnn_ensemble`` and produces a single wide DataFrame with one
row per sample per iteration, suitable for metric calculation and plotting.

Public API
----------
::

    from polynet.inference.gnn import get_predictions_df_gnn
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader

from polynet.config.column_names import (
    get_iterator_name,
    get_predicted_label_column_name,
    get_true_label_column_name,
)
from polynet.config.constants import DataSet, ResultColumn
from polynet.config.enums import ProblemType, SplitType
from polynet.inference.utils import assemble_predictions, prepare_probs_df

logger = logging.getLogger(__name__)


def get_predictions_df_gnn(
    models: dict,
    loaders: dict,
    problem_type: ProblemType | str,
    split_type: SplitType | str,
    target_variable_name: str | None = None,
) -> pd.DataFrame:
    """
    Collect GNN predictions across all models and iterations into a DataFrame.

    For each trained model, runs inference on the train, validation, and
    test loaders and assembles a wide DataFrame where multiple models from
    the same iteration appear as separate columns.

    Parameters
    ----------
    models:
        Dict of ``{"{arch}_{iteration}": fitted_model}`` as returned by
        ``train_gnn_ensemble``.
    loaders:
        Dict of ``{"{iteration}": (train_loader, val_loader, test_loader)}``
        as returned by ``train_gnn_ensemble``. These are prediction-only
        loaders (batch_size=1, no shuffle).
    problem_type:
        Classification or regression — determines whether probability
        columns are added.
    split_type:
        The split strategy used — determines the iterator column name.
    target_variable_name:
        Name of the target property used for column naming. If ``None``,
        generic names are used.

    Returns
    -------
    pd.DataFrame
        Wide predictions DataFrame with columns:
        - Sample index, set label (train/val/test), iterator, true labels
        - One predicted column per model
        - One probability column per class per model (classification only)
    """
    problem_type = ProblemType(problem_type) if isinstance(problem_type, str) else problem_type
    split_type = SplitType(split_type) if isinstance(split_type, str) else split_type

    label_col = get_true_label_column_name(target_variable_name)
    iterator = get_iterator_name(split_type)

    per_model_dfs: list[tuple[str, pd.DataFrame]] = []

    for model_name, model in models.items():
        gnn_arch, iteration = model_name.split("_")
        predicted_col = get_predicted_label_column_name(target_variable_name, gnn_arch)

        train_loader, val_loader, test_loader = loaders[iteration]

        # The training loader uses shuffle=True during training, so we
        # reconstruct it here with shuffle=False for deterministic inference.
        # Val and test loaders are already batch_size=1 and shuffle=False.
        train_loader = DataLoader(train_loader.dataset, batch_size=1, shuffle=False)

        splits = [
            (train_loader, DataSet.Training),
            (val_loader, DataSet.Validation),
            (test_loader, DataSet.Test),
        ]

        split_dfs: list[pd.DataFrame] = []

        for loader, set_label in splits:
            preds = model.predict_loader(loader)
            # Regression: predict_loader returns (idx, y_pred)
            # Classification: predict_loader returns (idx, y_pred, y_score)
            sample_ids = preds[0]
            y_pred = preds[1]
            y_true = np.concatenate([mol.y.cpu().detach().numpy() for mol in loader])

            split_df = pd.DataFrame(
                {
                    ResultColumn.INDEX: sample_ids,
                    ResultColumn.SET: set_label,
                    label_col: y_true,
                    predicted_col: y_pred,
                }
            )

            if problem_type == ProblemType.Classification:
                y_score = preds[2]
                probs_df = prepare_probs_df(
                    probs=y_score, target_variable_name=target_variable_name, model_name=gnn_arch
                )
                split_df[probs_df.columns] = probs_df.to_numpy()

            split_dfs.append(split_df)

        predictions_df = pd.concat(split_dfs, ignore_index=True)
        predictions_df[iterator] = iteration
        # Keep the last occurrence when the same sample appears in
        # multiple splits (can happen with LOO and overlapping sets)
        predictions_df = predictions_df[~predictions_df[ResultColumn.INDEX].duplicated(keep="last")]

        per_model_dfs.append((iteration, predictions_df))

    predictions = assemble_predictions(per_model_dfs, iterator, ResultColumn.INDEX)

    # Reorder: metadata columns first, then all prediction columns
    meta_cols = [ResultColumn.INDEX, ResultColumn.SET, iterator, label_col]
    pred_cols = [col for col in predictions.columns if col not in meta_cols]

    return predictions[meta_cols + pred_cols]
