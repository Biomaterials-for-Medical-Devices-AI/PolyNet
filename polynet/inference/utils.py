"""
polynet.inference.utils
========================
Shared utilities for assembling predictions DataFrames.

The iteration-grouping logic — merging per-model predictions from the same
iteration into a single wide DataFrame, then stacking iterations — is
shared by both GNN and TML inference and lives here as a private helper.

``prepare_probs_df`` generates standardised column names for per-class
probability scores, previously scattered across ``polynet.utils``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from polynet.config.column_names import get_score_column_name


def prepare_probs_df(probs: np.ndarray, target_variable_name: str, model_name: str) -> pd.DataFrame:
    """
    Build a DataFrame of per-class probability scores with standardised column names.

    Special case:
      - If probs has exactly 2 classes (shape (n_samples, 2)), return only the
        positive-class (class 1) probability column.
    """
    probs = np.asarray(probs)

    # Binary probabilities provided as (n_samples, 2): keep only class 1
    if probs.ndim == 2 and probs.shape[1] == 2:
        col = get_score_column_name(
            target_variable_name=target_variable_name, model_name=model_name, class_num=1
        )
        return pd.DataFrame({col: probs[:, 1]})

    # Binary probabilities already as (n_samples,)
    if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[1] == 1):
        col = get_score_column_name(
            target_variable_name=target_variable_name,
            model_name=model_name,
            class_num=1,  # treat as positive-class score
        )
        return pd.DataFrame({col: probs.reshape(-1)})

    # Multiclass: one column per class
    n_classes = probs.shape[1]
    columns = [
        get_score_column_name(
            target_variable_name=target_variable_name, model_name=model_name, class_num=c
        )
        for c in range(n_classes)
    ]
    return pd.DataFrame(probs, columns=columns)


def assemble_predictions(
    per_model_dfs: list[tuple[str, pd.DataFrame]], iterator: str, index_col: str
) -> pd.DataFrame:
    """
    Merge per-model prediction DataFrames into a single wide DataFrame.

    Within each iteration, predictions from multiple models are merged
    as new columns onto a shared index. Across iterations, the resulting
    DataFrames are stacked row-wise.

    Parameters
    ----------
    per_model_dfs:
        List of ``(iteration, predictions_df)`` tuples, one per model,
        in the order models were trained.
    iterator:
        Name of the iteration column (e.g. ``"bootstrap_iteration"``).
    index_col:
        Name of the sample index column used for deduplication.

    Returns
    -------
    pd.DataFrame
        Wide predictions DataFrame with one row per sample per iteration
        and one prediction column per model.
    """
    all_dfs: list[pd.DataFrame] = []
    current_df: pd.DataFrame | None = None
    current_iteration: str | None = None

    for iteration, predictions_df in per_model_dfs:
        if current_iteration is None:
            current_df = predictions_df.copy()
            current_iteration = iteration

        elif iteration == current_iteration:
            # Same iteration — merge new model columns onto existing rows
            new_cols = [col for col in predictions_df.columns if col not in current_df.columns]
            current_df = pd.concat([current_df, predictions_df[new_cols]], axis=1)

        else:
            # New iteration — save current and start fresh
            all_dfs.append(current_df)
            current_df = predictions_df.copy()
            current_iteration = iteration

    if current_df is not None:
        all_dfs.append(current_df)

    return pd.concat(all_dfs, ignore_index=False)
