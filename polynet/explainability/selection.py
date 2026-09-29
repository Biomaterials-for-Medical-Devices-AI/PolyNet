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
