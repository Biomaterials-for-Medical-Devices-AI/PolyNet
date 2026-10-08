"""
polynet.utils.validation
========================
Validators that check user settings against the data.

Pydantic schemas validate everything that can be decided from the config
alone (e.g. ``hpo_n_folds >= 2``). Some rules can only be checked once the
dataset has been loaded and split — for example, the number of
cross-validation folds cannot exceed the number of samples. Those checks live
here, free of any pipeline, GUI or training code, so that every entry point
(CLI, GUI, Python API) and every step that needs them can call the same
function before any training starts.

Public API
----------
::

    from polynet.utils.validation import (
        check_n_folds,
        check_n_folds_for_splits,
        validate_hpo_folds,
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from polynet.config.enums import HpoSplitStrategy, ProblemType
from polynet.config.schemas import DataConfig, TrainGNNConfig, TrainTMLConfig

# ---------------------------------------------------------------------------
# Cross-validation folds
# ---------------------------------------------------------------------------


def check_n_folds(n_folds: int, y, problem_type: ProblemType) -> None:
    """
    Check that ``n_folds`` is usable for cross-validation on targets ``y``.

    Parameters
    ----------
    n_folds:
        Requested number of folds ``k``.
    y:
        Targets of the samples the cross-validation will run on.
    problem_type:
        Classification or regression.

    Raises
    ------
    ValueError
        If ``k`` exceeds the number of samples or, for classification, the
        size of the smallest class (stratified folds need every class in
        every fold). The message states the allowed range.
    """
    y = pd.Series(np.asarray(y).ravel())
    n_samples = len(y)

    if n_folds > n_samples:
        raise ValueError(
            f"{n_folds} folds exceed the number of samples ({n_samples}). "
            f"Choose between 2 and {n_samples} folds."
        )
    if problem_type == ProblemType.Classification:
        class_counts = y.value_counts()
        smallest_class, smallest_count = class_counts.idxmin(), int(class_counts.min())
        if n_folds > smallest_count:
            raise ValueError(
                f"{n_folds} folds exceed the size of the smallest class (class {smallest_class} "
                f"has {smallest_count} samples). Stratified cross-validation needs every class "
                f"in every fold: choose between 2 and {smallest_count} folds."
            )


def check_n_folds_for_splits(
    n_folds: int,
    y: pd.Series,
    train_val_test_idxs: tuple,
    problem_type: ProblemType,
    setting_name: str = "hpo_n_folds",
    include_validation: bool = True,
) -> None:
    """
    Run ``check_n_folds`` on the hyperparameter-search samples of every split.

    Hyperparameter search runs on the training + validation samples of each
    outer split (or on the training samples only, for TML models with
    ``include_validation_in_training: false``), so ``k`` is checked against
    exactly those samples.

    Parameters
    ----------
    n_folds:
        Requested number of folds ``k``.
    y:
        Targets of the whole dataset, indexed like the split indices.
    train_val_test_idxs:
        ``(train_ids, val_ids, test_ids)``, each a list with one entry per split.
    problem_type:
        Classification or regression.
    setting_name:
        Config field reported in the error message (e.g. ``"tml_models.hpo_n_folds"``).
    include_validation:
        Whether the search samples include the validation samples.

    Raises
    ------
    ValueError
        If ``k`` is invalid for any split; the message names the setting and
        the split.
    """
    train_ids, val_ids, _ = train_val_test_idxs
    for i, (train_idxs, val_idxs) in enumerate(zip(train_ids, val_ids), start=1):
        hpo_idxs = pd.Index(train_idxs)
        if include_validation and val_idxs is not None:
            hpo_idxs = hpo_idxs.append(pd.Index(val_idxs))
        try:
            check_n_folds(n_folds=n_folds, y=y.loc[hpo_idxs], problem_type=problem_type)
        except ValueError as e:
            raise ValueError(f"{setting_name}={n_folds} is invalid for split {i}: {e}") from None


def validate_hpo_folds(
    data: pd.DataFrame,
    data_cfg: DataConfig,
    split_indexes: tuple,
    tml_cfg: TrainTMLConfig | None = None,
    gnn_cfg: TrainGNNConfig | None = None,
) -> None:
    """
    Check the HPO fold count of each pipeline against the data.

    Both pipelines score hyperparameter configurations by shuffled K-fold
    cross-validation on the training + validation samples of each split
    (TML: training samples only when ``include_validation_in_training`` is
    False; ``polynet.training.cv.make_kfold``). The schema already guarantees
    ``k >= 2``; this checks the rules that need the data. Call it after the
    data has been split and before any training starts.

    A pipeline is only checked when it will actually run cross-validated
    HPO: TML when at least one model has an empty hyperparameter block; GNN
    when at least one architecture has an empty block and
    ``hpo_split_strategy`` is ``cross_validation``.

    Parameters
    ----------
    data:
        The dataset that was split (its index labels are the split indices).
    data_cfg:
        Data configuration (target column, problem type).
    split_indexes:
        ``(train_idxs, val_idxs, test_idxs)`` from ``compute_data_splits``.
    tml_cfg:
        TML training configuration, or ``None`` if TML is not trained.
    gnn_cfg:
        GNN training configuration, or ``None`` if GNNs are not trained.

    Raises
    ------
    ValueError
        If ``hpo_n_folds`` is impossible for any split; the message names the
        config setting, the split and the allowed range.
    """
    checks = []
    if tml_cfg is not None and any(not p for p in (tml_cfg.selected_models or {}).values()):
        checks.append(
            ("tml_models.hpo_n_folds", tml_cfg.hpo_n_folds, tml_cfg.include_validation_in_training)
        )
    if (
        gnn_cfg is not None
        and gnn_cfg.hpo_split_strategy == HpoSplitStrategy.CrossValidation
        and any(not p for p in gnn_cfg.gnn_convolutional_layers.values())
    ):
        checks.append(("gnn_training.hpo_n_folds", gnn_cfg.hpo_n_folds, True))

    y = data[data_cfg.target_variable_col]
    for setting_name, n_folds, include_validation in checks:
        check_n_folds_for_splits(
            n_folds=n_folds,
            y=y,
            train_val_test_idxs=split_indexes,
            problem_type=data_cfg.problem_type,
            setting_name=setting_name,
            include_validation=include_validation,
        )


# ---------------------------------------------------------------------------
# Sample identifiers
# ---------------------------------------------------------------------------


def find_duplicate_ids(ids) -> list:
    """
    Return the identifier values that appear more than once.

    Parameters
    ----------
    ids:
        Sample identifiers (e.g. the ``id_col`` column).

    Returns
    -------
    list
        Each duplicated value once, in order of first appearance; empty when
        all identifiers are unique.
    """
    ids = pd.Series(ids)
    return ids[ids.duplicated(keep="first")].drop_duplicates().tolist()
