"""
polynet.training.cv
===================
Cross-validation helpers shared by the TML and GNN hyperparameter searches.

Both pipelines tune hyperparameters with K-fold cross-validation on the
training + validation samples of each outer split (TML always; GNN when
``hpo_split_strategy`` is ``cross_validation``). This module owns the fold
splitter they both use and the data-aware checks on the number of folds.

The static rule ``k >= 2`` lives in the config schema
(``HyperparamOptimConfig.hpo_n_folds``). The rules that depend on the data
(``k`` vs. number of samples and smallest class) can only be checked once the
dataset has been split, and are applied here.

Public API
----------
::

    from polynet.training.cv import make_kfold, check_n_folds, check_n_folds_for_splits
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from polynet.config.enums import ProblemType


def make_kfold(problem_type: ProblemType, n_folds: int, random_seed: int) -> KFold | StratifiedKFold:
    """
    Return the shuffled K-fold splitter used for hyperparameter search.

    Folds are always shuffled so that a dataset sorted by target (or by any
    other column) does not produce biased folds. Classification uses
    stratified folds to preserve class proportions.

    Parameters
    ----------
    problem_type:
        Classification or regression.
    n_folds:
        Number of folds ``k``.
    random_seed:
        Seed for the fold shuffling.

    Returns
    -------
    KFold | StratifiedKFold
        The splitter.
    """
    if problem_type == ProblemType.Classification:
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    return KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)


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
) -> None:
    """
    Run ``check_n_folds`` on the hyperparameter-search samples of every split.

    Hyperparameter search runs on the training + validation samples of each
    outer split, so ``k`` is checked against exactly those samples.

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

    Raises
    ------
    ValueError
        If ``k`` is invalid for any split; the message names the setting and
        the split.
    """
    train_ids, val_ids, _ = train_val_test_idxs
    for i, (train_idxs, val_idxs) in enumerate(zip(train_ids, val_ids), start=1):
        hpo_idxs = pd.Index(train_idxs).append(pd.Index(val_idxs if val_idxs is not None else []))
        try:
            check_n_folds(n_folds=n_folds, y=y.loc[hpo_idxs], problem_type=problem_type)
        except ValueError as e:
            raise ValueError(f"{setting_name}={n_folds} is invalid for split {i}: {e}") from None
