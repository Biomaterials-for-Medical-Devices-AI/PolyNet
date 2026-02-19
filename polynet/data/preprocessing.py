"""
polynet.data.preprocessing
===========================
Data preprocessing utilities for the polynet pipeline.

Covers three distinct preprocessing concerns:

1. **Class balancing** — undersampling the majority class for imbalanced
   classification datasets.
2. **Feature transformation** — scaling and transforming descriptor vectors
   before traditional ML training.
3. **DataFrame sanitisation** — dropping NaN columns, subsetting to
   relevant columns, extracting index metadata.

Public API
----------
::

    from polynet.data.preprocessing import (
        class_balancer,
        transform_features,
        sanitise_df,
        get_data_index,
    )
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd

from polynet.config.enums import TransformDescriptor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Class balancing
# ---------------------------------------------------------------------------


def class_balancer(
    data: pd.DataFrame, target: str, desired_class_proportion: float, random_state: int = 42
) -> pd.DataFrame:
    """
    Balance a binary classification dataset by undersampling the majority class.

    Parameters
    ----------
    data:
        Dataset containing the target column.
    target:
        Name of the binary target column.
    desired_class_proportion:
        Desired proportion of the minority class after balancing.
        For example, ``0.4`` means 40% minority / 60% majority.
        Must be in ``(0, 1)``.
    random_state:
        Random seed for reproducible undersampling.

    Returns
    -------
    pd.DataFrame
        A balanced copy of the dataset. The original DataFrame is not
        modified.

    Raises
    ------
    ValueError
        If the target column is not present, the dataset is None, the
        target is not binary, or ``desired_class_proportion`` is out of
        range.

    Examples
    --------
    >>> from polynet.data.preprocessing import class_balancer
    >>> balanced_df = class_balancer(df, target="activity", desired_class_proportion=0.4)
    """
    if data is None or target is None:
        raise ValueError("data and target must be provided.")

    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in the dataset.")

    if not (0 < desired_class_proportion < 1):
        raise ValueError(
            f"desired_class_proportion must be between 0 and 1, got {desired_class_proportion}."
        )

    class_counts = Counter(data[target])

    if len(class_counts) != 2:
        raise ValueError(
            f"class_balancer requires a binary target column, but '{target}' "
            f"has {len(class_counts)} unique values: {list(class_counts.keys())}."
        )

    minority_class, majority_class = sorted(class_counts, key=lambda x: class_counts[x])
    minority_size = class_counts[minority_class]
    majority_size = class_counts[majority_class]

    target_majority_size = int(
        minority_size * (1 - desired_class_proportion) / desired_class_proportion
    )
    samples_to_remove = majority_size - target_majority_size

    if samples_to_remove <= 0:
        logger.info(
            f"No balancing needed — dataset already meets the desired "
            f"proportion ({desired_class_proportion:.2f})."
        )
        return data

    logger.info(
        f"Class balancing: removing {samples_to_remove} samples from "
        f"majority class '{majority_class}'."
    )

    # Use numpy for reproducible sampling consistent with the rest of the pipeline
    rng = np.random.default_rng(random_state)
    majority_indices = data[data[target] == majority_class].index.tolist()
    indices_to_drop = rng.choice(majority_indices, size=samples_to_remove, replace=False)

    balanced = data.drop(index=indices_to_drop)

    _log_class_balance(balanced, target)
    return balanced


def _log_class_balance(data: pd.DataFrame, target: str) -> None:
    """Log the class distribution of a dataset."""
    counts = Counter(data[target])
    total = sum(counts.values())
    summary = "  ".join(
        f"[{cls}]: {count} ({count / total:.2f})" for cls, count in sorted(counts.items())
    )
    logger.info(f"Class balance after balancing → {summary}")


# ---------------------------------------------------------------------------
# Feature transformation
# ---------------------------------------------------------------------------


def transform_features(
    fit_data: pd.DataFrame, transform_data: pd.DataFrame, transform_type: TransformDescriptor | str
) -> tuple[np.ndarray, object]:
    """
    Fit a feature scaler on ``fit_data`` and apply it to ``transform_data``.

    The scaler is fitted on the training set and applied to any other set
    (validation, test) to prevent data leakage.

    Parameters
    ----------
    fit_data:
        DataFrame used to fit the scaler (typically the training set).
    transform_data:
        DataFrame to transform (may be the same as ``fit_data`` for
        the training set, or a held-out set).
    transform_type:
        The transformation to apply. Use ``TransformDescriptor.NoTransformation``
        to return the data unchanged.

    Returns
    -------
    tuple[np.ndarray, object]
        A tuple of ``(transformed_array, fitted_scaler)``. The scaler is
        returned so it can be saved and applied to future predictions.
        For ``NoTransformation``, the scaler is ``None``.

    Raises
    ------
    ValueError
        If ``transform_type`` is not recognised.

    Examples
    --------
    >>> from polynet.data.preprocessing import transform_features
    >>> from polynet.config.enums import TransformDescriptor
    >>> X_train_scaled, scaler = transform_features(X_train, X_train, TransformDescriptor.StandardScaler)
    >>> X_test_scaled, _ = transform_features(X_train, X_test, TransformDescriptor.StandardScaler)
    """
    from sklearn.preprocessing import (
        MinMaxScaler,
        Normalizer,
        PowerTransformer,
        QuantileTransformer,
        RobustScaler,
        StandardScaler,
    )

    transform_type = (
        TransformDescriptor(transform_type) if isinstance(transform_type, str) else transform_type
    )

    _SCALER_REGISTRY: dict[TransformDescriptor, object] = {
        TransformDescriptor.StandardScaler: StandardScaler(),
        TransformDescriptor.MinMaxScaler: MinMaxScaler(),
        TransformDescriptor.RobustScaler: RobustScaler(),
        TransformDescriptor.PowerTransformer: PowerTransformer(),
        TransformDescriptor.QuantileTransformer: QuantileTransformer(),
        TransformDescriptor.Normalizer: Normalizer(),
    }

    if transform_type == TransformDescriptor.NoTransformation:
        return transform_data.values if hasattr(transform_data, "values") else transform_data, None

    if transform_type not in _SCALER_REGISTRY:
        raise ValueError(
            f"Unsupported transformation type: '{transform_type}'. "
            f"Available: {[t.value for t in _SCALER_REGISTRY]}."
        )

    scaler = _SCALER_REGISTRY[transform_type]
    scaler.fit(fit_data)
    transformed = scaler.transform(transform_data)
    return transformed, scaler


# ---------------------------------------------------------------------------
# DataFrame utilities
# ---------------------------------------------------------------------------


def sanitise_df(df: pd.DataFrame, descriptors: list[str], target_variable_col: str) -> pd.DataFrame:
    """
    Subset a DataFrame to the selected descriptor columns and target,
    dropping any columns that contain NaN values.

    Parameters
    ----------
    df:
        Full dataset DataFrame.
    descriptors:
        List of descriptor column names to retain.
    target_variable_col:
        Name of the target column to retain.

    Returns
    -------
    pd.DataFrame
        A clean copy containing only the requested columns with no NaNs.

    Notes
    -----
    Columns are dropped, not rows — a descriptor column containing a single
    NaN is removed entirely. If row-wise dropping is preferred, apply
    ``df.dropna(axis=0)`` after calling this function.
    """
    cols = [c for c in descriptors if c in df.columns] + [target_variable_col]
    return df[cols].dropna(axis=1).copy()


def get_data_index(
    data: pd.DataFrame,
    id_col: str | None,
    smiles_cols: list[str],
    weights_col: dict[str, str] | None,
    target_col: str,
) -> pd.DataFrame:
    """
    Extract the metadata index columns from a dataset.

    Returns a DataFrame containing the identifier column, SMILES columns,
    weight columns (if any), and the target column. Used to attach
    provenance metadata to descriptor and prediction DataFrames throughout
    the pipeline.

    Parameters
    ----------
    data:
        Full dataset DataFrame.
    id_col:
        Optional sample identifier column name.
    smiles_cols:
        SMILES column names.
    weights_col:
        Optional mapping from SMILES column to weight fraction column.
    target_col:
        Target property column name.

    Returns
    -------
    pd.DataFrame
        A copy of the selected index columns.
    """
    idx_cols: list[str] = []

    if id_col and id_col in data.columns:
        idx_cols.append(id_col)

    idx_cols.extend(smiles_cols)

    if weights_col:
        weight_cols = [c for c in weights_col.values() if c in data.columns]
        idx_cols.extend(weight_cols)

    if target_col in data.columns:
        idx_cols.append(target_col)

    return data[idx_cols].copy()
