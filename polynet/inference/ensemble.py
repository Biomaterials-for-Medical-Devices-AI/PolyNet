"""
polynet.inference.ensemble
==========================
Ensemble predictions across the models trained on the repeated random splits.

A PolyNet experiment trains one model per split for every architecture (GNN)
or model × representation (TML). When predicting new data, the per-split
predictions of the same model are combined into an ensemble:

- **Regression** — the ensemble prediction is the mean of the member
  predictions, reported together with their standard deviation (the spread
  between members; not a calibrated uncertainty).
- **Classification** — the ensemble prediction is the majority vote, reported
  together with the vote fraction (the share of members that voted for the
  ensemble class).

::

    from polynet.inference.ensemble import ensemble_predictions
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import mode

from polynet.config.column_names import get_predicted_label_column_name
from polynet.config.enums import ProblemType

ENSEMBLE = "Ensemble"
ENSEMBLE_STD = "Ensemble Std"
ENSEMBLE_VOTE_FRACTION = "Ensemble Vote Fraction"


def ensemble_model_name(group_name: str) -> str:
    """Model name used for an ensemble in column names and metrics, e.g. ``"GCN Ensemble"``."""
    return f"{group_name} {ENSEMBLE}"


def ensemble_predictions(
    predictions: pd.DataFrame,
    groups: dict[str, list[str]],
    problem_type: ProblemType | str,
    target_variable_name: str | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Combine per-split prediction columns into ensemble columns.

    Parameters
    ----------
    predictions:
        DataFrame holding the per-split predicted-value columns.
    groups:
        Mapping from ensemble name (e.g. ``"GCN"``, ``"random forest-rdkit"``)
        to the predicted-value columns of its members, one per split. Groups
        with fewer than two members are skipped (there is nothing to combine).
    problem_type:
        Classification or regression.
    target_variable_name:
        Human-readable target name, used to build the column names.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, str]]
        ``(ensemble_df, predicted_cols)``:

        - ``ensemble_df`` has, per group, the ensemble prediction column
          (``"{name} Ensemble Predicted {target}"``) plus
          ``"{name} Ensemble Std {target}"`` (population standard deviation of
          the member predictions, regression) or
          ``"{name} Ensemble Vote Fraction {target}"`` (share of members that
          voted for the ensemble class, classification). It shares the index
          of ``predictions``.
        - ``predicted_cols`` maps each ensemble prediction column to its
          ensemble model name (``"{name} Ensemble"``), for metric computation.

    Notes
    -----
    Classification ties are broken towards the smallest class label
    (``scipy.stats.mode``).
    """
    problem_type = ProblemType(problem_type)
    columns: dict[str, pd.Series] = {}
    predicted_cols: dict[str, str] = {}

    for name, cols in groups.items():
        if len(cols) < 2:
            continue
        member_preds = predictions[cols]
        model_name = ensemble_model_name(name)
        pred_col = get_predicted_label_column_name(
            target_variable_name=target_variable_name, model_name=model_name
        )
        suffix = f" {target_variable_name}" if target_variable_name else ""

        if problem_type == ProblemType.Classification:
            votes, _ = mode(member_preds.to_numpy(), axis=1, keepdims=False)
            columns[pred_col] = pd.Series(votes, index=member_preds.index)
            columns[f"{name} {ENSEMBLE_VOTE_FRACTION}{suffix}"] = member_preds.eq(
                columns[pred_col], axis=0
            ).mean(axis=1)
        else:
            columns[pred_col] = member_preds.mean(axis=1)
            columns[f"{name} {ENSEMBLE_STD}{suffix}"] = member_preds.std(axis=1, ddof=0)

        predicted_cols[pred_col] = model_name

    return pd.DataFrame(columns, index=predictions.index), predicted_cols
