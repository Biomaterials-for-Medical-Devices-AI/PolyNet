"""
polynet.explainability.selection
================================
Which trained models and which samples an explanation covers.

Shared by the GNN and TML explainability stages (and usable by the GUI):

- ``select_splits`` resolves the ``bootstraps`` setting — split numbers
  counted from 1, like the model names (``GCN_1``), ``metrics.json`` and the
  ``bootstrap_iteration`` column;
- ``split_index_of_model`` reads the split of a model from its name;
- ``match_dataset_ids`` maps requested sample IDs (strings from configs and
  split files) to the IDs stored on the graphs, whatever the ID column type.

::

    from polynet.explainability.selection import (
        match_dataset_ids,
        select_splits,
        split_index_of_model,
    )
"""

from __future__ import annotations

import logging

from polynet.config.constants import DataSet, ResultColumn

logger = logging.getLogger(__name__)


def select_splits(bootstraps, n_splits: int) -> list[int]:
    """
    Resolve the ``bootstraps`` setting of an explainability config.

    ``bootstraps`` holds split numbers counted from 1 (or ``"all"``), the same
    numbers used in model names (``GCN_1``), ``metrics.json`` and the
    ``bootstrap_iteration`` column.

    Returns
    -------
    list[int]
        Sorted 0-based positions in the split-index lists (split 1 → 0);
        numbers outside ``1..n_splits`` are dropped with a warning.
    """
    if bootstraps == "all":
        return list(range(n_splits))
    numbers = sorted({b for b in bootstraps if 1 <= b <= n_splits})
    invalid = sorted(set(bootstraps) - set(numbers))
    if invalid:
        logger.warning(
            f"Bootstrap numbers {invalid} are out of range (splits are numbered 1–{n_splits}). "
            "They will be skipped."
        )
    return [b - 1 for b in numbers]


def split_index_of_model(model_key: str) -> int:
    """0-based split index of a model named ``{name}_{split number}`` (e.g. ``GCN_1`` → 0)."""
    return int(model_key.rsplit("_", 1)[1]) - 1


def match_dataset_ids(dataset, ids: list, what: str) -> list:
    """
    Map requested sample IDs to the IDs stored on the graphs of ``dataset``.

    Config and split IDs are strings, while graph IDs keep the type of the ID
    column (e.g. ``int``); matching is done on the string form so both work.
    IDs not in the dataset are dropped with a warning.
    """
    by_str = {str(graph.idx): graph.idx for graph in dataset}
    missing = [i for i in ids if str(i) not in by_str]
    if missing:
        logger.warning(f"{len(missing)} {what} ID(s) not found in the dataset, e.g. {missing[:5]}.")
    return [by_str[str(i)] for i in ids if str(i) in by_str]


# Explain-set names used by the CLI configs and by the GUI (DataSet labels).
_SET_POSITIONS = {"train": 0, "validation": 1, "test": 2}
_GUI_SET_NAMES = {DataSet.Training, DataSet.Validation, DataSet.Test}

TML_VALIDATION_IS_TRAINING_WARNING = (
    "Traditional ML models are trained on the training and validation samples "
    "together, so validation samples are not held out for these models: their SHAP "
    "values describe data the model was fitted on. Use the test set to explain "
    "predictions on unseen samples."
)


def tml_explain_set_includes_validation(explain_set: str | None) -> bool:
    """
    Whether a TML explain set contains validation samples.

    Parameters
    ----------
    explain_set : str or None
        CLI name (``"train"``, ``"validation"``, ``"test"``, ``"all"``) or GUI
        label (``DataSet.Validation``, ``"All"``, ...).

    Returns
    -------
    bool
        True for the validation set and for ``"all"``; these samples were part
        of TML training.
    """
    return explain_set is not None and str(explain_set).lower() in {"validation", "all"}


def samples_per_model(model_keys, split_indexes: tuple, explain_set: str) -> dict[str, set[str]]:
    """
    Samples each model may explain in a global (population) explanation.

    Each model was trained on one split; a global explanation of the ``test``
    (or ``train`` / ``validation``) set should only use that split's samples,
    otherwise a model would "explain" molecules that were in its own training
    data. ``explain_set="all"`` allows every sample of the model's split.

    Parameters
    ----------
    model_keys:
        Model names ending with the 1-based split number (``GCN_1``,
        ``random_forest-rdkit_2``).
    split_indexes:
        ``(train_idxs, val_idxs, test_idxs)``, one list of sample IDs per split.
    explain_set:
        ``"train"``, ``"validation"``, ``"test"`` or ``"all"``.

    Returns
    -------
    dict[str, set[str]]
        ``{model_key: {sample IDs as strings}}``.
    """
    positions = (
        list(_SET_POSITIONS.values()) if explain_set == "all" else [_SET_POSITIONS[explain_set]]
    )
    allowed = {}
    for key in model_keys:
        split = split_index_of_model(key)
        allowed[key] = {
            str(i) for p in positions for i in (split_indexes[p][split] if split_indexes[p] else [])
        }
    return allowed


def samples_per_model_from_predictions(
    predictions, model_keys, iterator_col: str | None, set_name: str | None
) -> dict[str, set[str]] | None:
    """
    GUI version of ``samples_per_model``, using the predictions table.

    Parameters
    ----------
    predictions:
        Predictions indexed by sample ID, one row per (sample × split), with
        the split number in ``iterator_col`` and the set in ``ResultColumn.SET``.
    model_keys:
        Selected model names ending with the split number.
    iterator_col:
        Name of the split-number column (``None`` for external data).
    set_name:
        ``"Training"``, ``"Validation"`` or ``"Test"``; anything else (``"All"``,
        ``"External"``, no choice) means no per-model restriction.

    Returns
    -------
    dict[str, set[str]] | None
        ``{model_key: {sample IDs as strings}}``, or ``None`` when every
        selected sample may be explained by every model.
    """
    if set_name not in _GUI_SET_NAMES or not iterator_col or iterator_col not in predictions:
        return None
    in_set = predictions[predictions[ResultColumn.SET] == set_name]
    return {
        key: {str(i) for i in in_set.index[in_set[iterator_col] == split_index_of_model(key) + 1]}
        for key in model_keys
    }
