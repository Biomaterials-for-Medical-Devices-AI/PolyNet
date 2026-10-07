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

import logging
import math
from typing import Generator
import warnings

from astartes import train_val_test_split
from astartes.utils.exceptions import InvalidConfigurationError
from astartes.utils.warnings import ImperfectSplittingWarning, NormalizationWarning
import numpy as np
import pandas as pd
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from polynet.config.enums import SplitMethod, SplitSampler, SplitType
from polynet.config.schemas.split_data import (
    SAMPLERS_USING_FINGERPRINTS,
    SAMPLERS_USING_TARGET,
    stratified_target_sampler_error,
)
from polynet.data.preprocessing import class_balancer

logger = logging.getLogger(__name__)

# Samplers that fill the sets by count (floor rounding only), so astartes'
# size warning only reports rounding to whole samples. Cluster samplers keep
# clusters whole and can miss the requested sizes; that warning is logged.
_COUNT_FILLED_SAMPLERS = frozenset(
    {
        SplitSampler.Random,
        SplitSampler.KennardStone,
        SplitSampler.SPXY,
        SplitSampler.TargetProperty,
    }
)

# ---------------------------------------------------------------------------
# Index computation
# ---------------------------------------------------------------------------


def _split_fractions(val_ratio: float, test_ratio: float) -> tuple[float, float, float]:
    """
    Train / validation / test fractions that sum to exactly 1.0.

    astartes rescales fractions whose sum is not exactly 1.0, so the training
    fraction is nudged by a few floating-point steps until ``train + test +
    val == 1.0``. For some ratio pairs no such value exists; astartes then
    rescales by 1 ± 1e-16, which does not change the split sizes.
    """
    train = 1.0 - val_ratio - test_ratio
    up = down = train
    for _ in range(4):
        for candidate in (up, down):
            if candidate + test_ratio + val_ratio == 1.0:
                return candidate, val_ratio, test_ratio
        up, down = math.nextafter(up, 1.0), math.nextafter(down, 0.0)
    return train, val_ratio, test_ratio


def astartes_split(
    n_samples: int,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
    sampler: SplitSampler | str = SplitSampler.Random,
    features: np.ndarray | None = None,
    targets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Train / validation / test split of ``n_samples`` with an astartes sampler.

    Uses ``astartes.train_val_test_split`` with astartes' default sampler
    hyperparameters.

    - ``random``: a random split (``features`` are not needed). Training and
      validation sizes are ``floor(n × fraction)``; test takes the rest.
    - Fingerprint samplers (``kennard_stone``, ``spxy``, ``kmeans``,
      ``optisim``) place samples by their ``features``; ``spxy`` also uses
      the ``targets``.
    - ``target_property`` orders samples by their ``targets``.

    Cluster-based samplers keep each cluster in one set, so the validation
    and test sets can be smaller than requested; astartes' warning about it
    is logged.

    Parameters
    ----------
    n_samples:
        Number of samples to split.
    val_ratio:
        Fraction of the samples for validation.
    test_ratio:
        Fraction of the samples for testing.
    random_seed:
        Seed of the sampler (ignored by the deterministic samplers).
    sampler:
        The astartes sampler.
    features:
        Array of shape ``(n_samples, n_features)``; required by the
        fingerprint samplers.
    targets:
        Target values of the samples; required by ``spxy`` and
        ``target_property``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Positions (``0 … n_samples − 1``) of the training, validation and
        test samples.

    Raises
    ------
    ValueError
        If required inputs are missing, or the sampler cannot fill non-empty
        training, validation and test sets.
    """
    sampler = SplitSampler(sampler)
    if sampler in SAMPLERS_USING_FINGERPRINTS and features is None:
        raise ValueError(f"The '{sampler.value}' sampler needs fingerprint features.")
    if sampler in SAMPLERS_USING_TARGET and targets is None:
        raise ValueError(f"The '{sampler.value}' sampler needs the target values.")

    X = features if sampler in SAMPLERS_USING_FINGERPRINTS else np.arange(n_samples).reshape(-1, 1)
    y = np.asarray(targets, dtype=float) if sampler in SAMPLERS_USING_TARGET else None
    train_size, val_size, test_size = _split_fractions(val_ratio, test_ratio)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            *_, train_idx, val_idx, test_idx = train_val_test_split(
                X,
                y=y,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
                sampler=sampler.value,
                random_state=random_seed,
                hopts={},
                return_indices=True,
            )
    except InvalidConfigurationError as exc:
        raise ValueError(
            f"The '{sampler.value}' sampler cannot split {n_samples} samples into training, "
            f"validation ({val_ratio}) and test ({test_ratio}) sets: {exc}"
        ) from exc

    for w in caught:
        # Rounding to whole samples and the 1e-16 rescaling of
        # _split_fractions are expected; cluster-based sizes are reported.
        if issubclass(w.category, NormalizationWarning):
            continue
        if issubclass(w.category, ImperfectSplittingWarning) and sampler in _COUNT_FILLED_SAMPLERS:
            continue
        logger.warning(f"astartes ({sampler.value}): {w.message}")

    sets = [np.asarray(idx, dtype=int) for idx in (train_idx, val_idx, test_idx)]
    combined = np.concatenate(sets)
    if len(np.unique(combined)) != len(combined):
        # Guard against silent sampler failures (e.g. NaN distances make
        # Kennard–Stone-type samplers repeat one sample).
        raise ValueError(
            f"The '{sampler.value}' sampler returned repeated or overlapping samples; the "
            "split is not valid. Check the sampling fingerprints and targets."
        )
    return tuple(sets)


def stratified_astartes_split(
    labels: pd.Series,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
    sampler: SplitSampler | str = SplitSampler.Random,
    features: np.ndarray | None = None,
    targets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train / validation / test split built from astartes splits.

    The samples of each class are split separately with ``astartes_split``
    (same sampler, fractions and seed) and the results are merged, so every
    set keeps the class proportions of ``labels`` up to rounding within each
    class.

    Parameters
    ----------
    labels:
        Class label of every sample.
    val_ratio, test_ratio, random_seed, sampler, features, targets:
        As in ``astartes_split`` (``features`` / ``targets`` for all samples).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Positions of the training, validation and test samples.

    Raises
    ------
    ValueError
        If a class cannot be split into non-empty training, validation and
        test sets, or with a sampler that uses the target (``spxy``,
        ``target_property``): within a class the target is constant.
    """
    sampler = SplitSampler(sampler)
    if sampler in SAMPLERS_USING_TARGET:
        raise ValueError(stratified_target_sampler_error(sampler))
    values = np.asarray(labels)
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for label in sorted(pd.unique(values)):
        class_positions = np.flatnonzero(values == label)
        try:
            split = astartes_split(
                len(class_positions),
                val_ratio,
                test_ratio,
                random_seed,
                sampler=sampler,
                features=None if features is None else features[class_positions],
                targets=None if targets is None else np.asarray(targets)[class_positions],
            )
        except ValueError as exc:
            raise ValueError(
                f"Stratified split: class {getattr(label, 'item', lambda: label)()} "
                f"({len(class_positions)} sample(s)) cannot be split into training, validation "
                f"and test sets: {exc}"
            ) from exc
        parts.append(tuple(class_positions[idx] for idx in split))
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))


def _balance(
    data: pd.DataFrame, target: str, train_set_balance: float, random_seed: int, name: str
) -> pd.DataFrame:
    """Undersample the majority class of one set (left unchanged if it has a single class)."""
    if data[target].nunique() < 2:
        logger.warning(f"The {name} set has a single class; it is not balanced.")
        return data
    return class_balancer(
        data=data,
        target=target,
        desired_class_proportion=train_set_balance,
        random_state=random_seed,
    )


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
    sampler: SplitSampler | str = SplitSampler.Random,
    sampling_features: np.ndarray | None = None,
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
        Desired proportion of the minority class after undersampling the
        majority class (e.g. ``0.5`` for 50/50). ``None`` or ``1.0`` disables
        balancing. Only meaningful for binary classification. The training
        and validation sets are each balanced after the split, so only the
        test set keeps the original class distribution (see
        ``_train_val_test_indices``).
    random_seed:
        Base random seed. Each bootstrap iteration uses ``random_seed + i``
        to ensure reproducibility while varying the split.
    sampler:
        astartes sampler (see ``astartes_split``).
    sampling_features:
        Fingerprints of the rows of ``data`` (same order), required by the
        fingerprint samplers.

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
                sampler=SplitSampler(sampler),
                sampling_features=sampling_features,
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
    sampler: SplitSampler = SplitSampler.Random,
    sampling_features: np.ndarray | None = None,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """
    Compute indices for TrainValTest with repeated splits.

    For each iteration (seed ``random_seed + i``):

    1. split the full dataset into training, validation and test sets with
       the astartes ``sampler`` (per class for ``Stratified``), with
       ``val_ratio`` and ``test_ratio`` as fractions of the full dataset;
    2. optionally balance the training and the validation set, each by
       undersampling its majority class to ``train_set_balance``.

    As a result, training and validation sets are both balanced, while the
    test set keeps the original class distribution. This follows the
    protocol of ACS Appl. Mater. Interfaces 2023, 15 (11), 14155–14163, and is intentional.
    """

    train_data_idxs: list[list[int]] = []
    val_data_idxs: list[list[int]] = []
    test_data_idxs: list[list[int]] = []

    balance = train_set_balance is not None and train_set_balance < 1.0

    for i in range(n_bootstrap_iterations):
        seed = random_seed + i

        sampler_inputs = dict(
            sampler=sampler,
            features=sampling_features,
            targets=data[target_variable_col] if sampler in SAMPLERS_USING_TARGET else None,
        )
        if split_method == SplitMethod.Stratified:
            positions = stratified_astartes_split(
                data[target_variable_col], val_ratio, test_ratio, seed, **sampler_inputs
            )
        else:
            positions = astartes_split(len(data), val_ratio, test_ratio, seed, **sampler_inputs)
        train_data, val_data, test_data = (data.iloc[p] for p in positions)

        # Balance training and validation separately; test keeps the original
        # class distribution (see the docstring).
        if balance:
            train_data = _balance(train_data, target_variable_col, train_set_balance, seed, "training")
            val_data = _balance(val_data, target_variable_col, train_set_balance, seed, "validation")

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
