"""
polynet.training.cv
===================
Cross-validation helpers shared by the TML and GNN hyperparameter searches.

Both pipelines tune hyperparameters with K-fold cross-validation on the
training + validation samples of each outer split (TML always; GNN when
``hpo_split_strategy`` is ``cross_validation``). This module owns the fold
splitter they both use.

The number of folds is validated elsewhere: ``k >= 2`` in the config schema
(``HyperparamOptimConfig.hpo_n_folds``), and ``k`` against the data in
``polynet.utils.validation.validate_hpo_folds``.

Public API
----------
::

    from polynet.training.cv import make_kfold
"""

from __future__ import annotations

from sklearn.model_selection import KFold, StratifiedKFold

from polynet.config.enums import ProblemType


def make_kfold(
    problem_type: ProblemType, n_folds: int, random_seed: int
) -> KFold | StratifiedKFold:
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
