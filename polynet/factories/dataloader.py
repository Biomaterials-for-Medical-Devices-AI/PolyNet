"""
polynet.factories.dataloader
=============================
Index computation and DataLoader construction for all split strategies.

Two concerns are handled here:

1. **Index computation** — ``get_data_split_indices`` takes a DataFrame and
   returns lists of train/val/test indices for each iteration. This is
   dataset-agnostic and works with both GNN and TML pipelines.

2. **DataLoader construction** — ``SplitGenerator`` takes a PyG dataset and
   pre-computed index lists, and yields ``(train_loader, val_loader,
   test_loader)`` tuples for each iteration.

Note
----
``get_data_split_indices`` will migrate to ``polynet.data.splitter`` when
that module is implemented. It lives here for now to keep the factory layer
self-contained during the refactor.

Implementation status
---------------------
✅ TrainValTest (with bootstrap iterations and optional class balancing)
✅ LeaveOneOut
🔲 TrainTest
🔲 CrossValidation
🔲 NestedCrossValidation

Public API
----------
::

    from polynet.factories.dataloader import get_data_split_indices, SplitGenerator
    from polynet.config.enums import SplitType, SplitMethod

    train_idxs, val_idxs, test_idxs = get_data_split_indices(
        data=df,
        split_type=SplitType.TrainValTest,
        n_bootstrap_iterations=5,
        val_ratio=0.1,
        test_ratio=0.2,
        target_variable_col="Tg",
        split_method=SplitMethod.Random,
        train_set_balance=1.0,
        random_seed=42,
    )

    generator = SplitGenerator(
        split_type=SplitType.TrainValTest,
        batch_size=32,
    )

    for train_loader, val_loader, test_loader in generator.split(
        dataset,
        train_indices=train_idxs,
        val_indices=val_idxs,
        test_indices=test_idxs,
    ):
        ...
"""

from __future__ import annotations

from typing import Generator

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from polynet.config.enums import SplitMethod, SplitType
from polynet.data.preprocessing import class_balancer

# ---------------------------------------------------------------------------
# Index computation
# ---------------------------------------------------------------------------


def _raw_split(
    data: pd.DataFrame, test_size: float, random_state: int, stratify: pd.Series | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Thin wrapper around ``sklearn.train_test_split`` that preserves the
    original DataFrame index, which is used to track sample identity
    throughout the pipeline.
    """
    return train_test_split(data, test_size=test_size, random_state=random_state, stratify=stratify)


def get_data_split_indices(
    data: pd.DataFrame,
    split_type: SplitType | str,
    n_bootstrap_iterations: int,
    val_ratio: float,
    test_ratio: float,
    target_variable_col: str,
    split_method: SplitMethod | str,
    train_set_balance: float,
    random_seed: int,
) -> tuple[list[list[int]], list[list[int]] | None, list[list[int]]]:
    """
    Compute train / val / test index lists for each iteration of a split.

    Parameters
    ----------
    data:
        The full dataset as a DataFrame. Only the index is used — feature
        columns and target values are not read here.
    split_type:
        The splitting strategy to apply.
    n_bootstrap_iterations:
        Number of times to repeat the split with a different random seed
        offset. Only used for ``TrainValTest`` and ``TrainTest``.
    val_ratio:
        Fraction of the full dataset reserved for validation.
        Only used when ``split_type`` is ``TrainValTest``.
    test_ratio:
        Fraction of the full dataset reserved for testing.
    target_variable_col:
        Column name of the target variable. Used for stratified splitting.
    split_method:
        Whether to split randomly or stratified by the target variable.
    train_set_balance:
        Fraction of the training set to retain after class balancing.
        A value of ``1.0`` disables balancing. Only meaningful for
        classification tasks.
    random_seed:
        Base random seed. Each bootstrap iteration uses ``random_seed + i``
        to ensure reproducibility while varying the split.

    Returns
    -------
    tuple[list[list[int]], list[list[int]] | None, list[list[int]]]
        A triple of ``(train_indices, val_indices, test_indices)``.
        Each element is a list of lists — one inner list per iteration.
        ``val_indices`` is ``None`` for split types without a validation set.

    Raises
    ------
    NotImplementedError
        For split types that are not yet implemented.
    ValueError
        If ``split_type`` is not recognised.
    """
    split_type = SplitType(split_type) if isinstance(split_type, str) else split_type
    split_method = SplitMethod(split_method) if isinstance(split_method, str) else split_method

    match split_type:
        case SplitType.TrainValTest:
            return _train_val_test_indices(
                data=data,
                n_bootstrap_iterations=n_bootstrap_iterations,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                target_variable_col=target_variable_col,
                split_method=split_method,
                train_set_balance=train_set_balance,
                random_seed=random_seed,
            )

        case SplitType.LeaveOneOut:
            return _leave_one_out_indices(data)

        case SplitType.TrainTest:
            raise NotImplementedError(
                "TrainTest split index computation is not yet implemented. "
                "Use TrainValTest or LeaveOneOut instead."
            )

        case SplitType.CrossValidation:
            raise NotImplementedError(
                "CrossValidation split index computation is not yet implemented. "
                "Use TrainValTest or LeaveOneOut instead."
            )

        case SplitType.NestedCrossValidation:
            raise NotImplementedError(
                "NestedCrossValidation split index computation is not yet implemented. "
                "Use TrainValTest or LeaveOneOut instead."
            )

        case _:
            raise ValueError(
                f"Unrecognised split type: '{split_type}'. "
                f"Available: {[s.value for s in SplitType]}."
            )


def _train_val_test_indices(
    data: pd.DataFrame,
    n_bootstrap_iterations: int,
    val_ratio: float,
    test_ratio: float,
    target_variable_col: str,
    split_method: SplitMethod,
    train_set_balance: float | None,
    random_seed: int,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Compute indices for TrainValTest with bootstrap repetitions."""

    train_data_idxs: list[list[int]] = []
    val_data_idxs: list[list[int]] = []
    test_data_idxs: list[list[int]] = []

    use_stratify = split_method == SplitMethod.Stratified

    for i in range(n_bootstrap_iterations):
        seed = random_seed + i

        # Step 1: hold out test set
        train_data, test_data = _raw_split(
            data=data,
            test_size=test_ratio,
            random_state=seed,
            stratify=data[target_variable_col] if use_stratify else None,
        )

        # Step 2: optional class balancing on training data
        if train_set_balance is not None and train_set_balance < 1.0:
            train_data = class_balancer(
                data=train_data,
                target=target_variable_col,
                desired_class_proportion=train_set_balance,
                random_state=seed,
            )

        # Step 3: carve validation set out of training data
        train_data, val_data = _raw_split(
            data=train_data,
            test_size=val_ratio,
            random_state=seed,
            stratify=train_data[target_variable_col] if use_stratify else None,
        )

        train_data_idxs.append(train_data.index)
        val_data_idxs.append(val_data.index)
        test_data_idxs.append(test_data.index)

    return train_data_idxs, val_data_idxs, test_data_idxs


def _leave_one_out_indices(data: pd.DataFrame) -> tuple[list[list[int]], None, list[list[int]]]:
    """Compute indices for LeaveOneOut — each sample is the test set once."""
    n = len(data)
    all_indices = list(range(n))
    train_idxs = [[x for x in all_indices if x != i] for i in all_indices]
    test_idxs = [[i] for i in all_indices]
    return train_idxs, None, test_idxs


# ---------------------------------------------------------------------------
# DataLoader construction
# ---------------------------------------------------------------------------


class SplitGenerator:
    """
    Converts pre-computed index lists into PyG DataLoader tuples.

    Accepts the index lists returned by ``get_data_split_indices`` and
    yields one ``(train_loader, val_loader, test_loader)`` tuple per
    iteration. ``val_loader`` is ``None`` when val indices are not provided.

    Parameters
    ----------
    split_type:
        The splitting strategy. Used for context in error messages.
    batch_size:
        Batch size for the training DataLoader. Val and test loaders
        always use a batch size of 1.

    Examples
    --------
    >>> generator = SplitGenerator(SplitType.TrainValTest, batch_size=32)
    >>> for train_loader, val_loader, test_loader in generator.split(
    ...     dataset,
    ...     train_indices=train_idxs,
    ...     val_indices=val_idxs,
    ...     test_indices=test_idxs,
    ... ):
    ...     ...
    """

    def __init__(self, split_type: SplitType | str, batch_size: int) -> None:
        self.split_type = SplitType(split_type) if isinstance(split_type, str) else split_type
        self.batch_size = batch_size

    def split(
        self,
        dataset,
        *,
        train_indices: list[list[int]],
        test_indices: list[list[int]],
        val_indices: list[list[int]] | None = None,
    ) -> Generator[tuple[DataLoader, DataLoader | None, DataLoader], None, None]:
        """
        Yield ``(train_loader, val_loader, test_loader)`` for each iteration.

        Parameters
        ----------
        dataset:
            A PyG dataset or any object supporting integer indexing.
        train_indices:
            List of train index lists — one per iteration.
        test_indices:
            List of test index lists — one per iteration.
        val_indices:
            Optional list of val index lists — one per iteration.
            If ``None``, ``val_loader`` is ``None`` in every yielded tuple.

        Yields
        ------
        tuple[DataLoader, DataLoader | None, DataLoader]
        """
        for i in range(len(test_indices)):
            train_loader = self._make_loader(dataset, train_indices[i], shuffle=True)
            test_loader = self._make_loader(dataset, test_indices[i])
            val_loader = (
                self._make_loader(dataset, val_indices[i]) if val_indices is not None else None
            )
            yield train_loader, val_loader, test_loader

    def _make_loader(self, dataset, indices: list[int], *, shuffle: bool = False) -> DataLoader:
        """Construct a DataLoader from a dataset and an index list."""
        subset = Subset(dataset, indices)
        batch_size = self.batch_size if shuffle else 1
        return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
