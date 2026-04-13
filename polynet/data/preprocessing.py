"""
polynet.data.preprocessing
===========================
Data preprocessing utilities for the polynet pipeline.

Covers three distinct preprocessing concerns:

1. **Class balancing** — undersampling the majority class for imbalanced
   classification datasets.
2. **Target variable scaling** — fitting and applying a ``TargetScaler``
   to regression targets before training, with inverse-transform support.
3. **DataFrame sanitisation** — dropping NaN columns, subsetting to
   relevant columns, extracting index metadata.

Note: feature (X) scaling is handled by ``FeatureTransformer`` in
``polynet.data.feature_transformer``.

Public API
----------
::

    from polynet.data.preprocessing import (
        class_balancer,
        TargetScaler,
        sanitise_df,
        get_data_index,
    )
"""

from __future__ import annotations

from collections import Counter
import logging

import numpy as np
import pandas as pd

from polynet.config.enums import TargetTransformDescriptor

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
# DataFrame utilities
# ---------------------------------------------------------------------------


def sanitise_df(
    df: pd.DataFrame,
    smiles_cols: list[str],
    target_variable_col: str,
    weights_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Remove SMILES and optional weight columns, and ensure the target
    column is the last column in the DataFrame.

    Parameters
    ----------
    df:
        Full dataset DataFrame.
    smiles_cols:
        Column names containing SMILES strings to remove.
    target_variable_col:
        Name of the target column.
    weights_cols:
        Optional list of weight column names to remove.

    Returns
    -------
    pd.DataFrame
        A cleaned copy of the DataFrame with target column last.
    """
    df_clean = df.copy()

    # Columns to remove
    cols_to_drop = list(smiles_cols)

    if weights_cols is not None:
        cols_to_drop.extend(weights_cols)

    # Drop only columns that exist (avoid KeyError)
    cols_to_drop = [c for c in cols_to_drop if c in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_to_drop)

    if target_variable_col not in df_clean.columns:
        raise ValueError(f"Target column '{target_variable_col}' not found after sanitisation.")

    # Move target column to the end
    cols = [c for c in df_clean.columns if c != target_variable_col]
    cols.append(target_variable_col)

    return df_clean[cols]


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


# ---------------------------------------------------------------------------
# Target variable scaling
# ---------------------------------------------------------------------------


class TargetScaler:
    """
    Scaler for regression target variables.

    Fits exclusively on training targets and provides consistent
    ``transform`` / ``inverse_transform`` methods used to recover
    the original value range before metrics and plots are computed.

    Parameters
    ----------
    strategy:
        Scaling strategy. ``TargetTransformDescriptor.NoTransformation``
        (the default) leaves values unchanged.

    Examples
    --------
    >>> from polynet.data.preprocessing import TargetScaler
    >>> from polynet.config.enums import TargetTransformDescriptor
    >>> scaler = TargetScaler(TargetTransformDescriptor.StandardScaler)
    >>> scaler.fit(y_train)
    >>> y_train_scaled = scaler.transform(y_train)
    >>> y_pred_original = scaler.inverse_transform(y_pred_scaled)
    """

    _SKLEARN_STRATEGIES = {
        TargetTransformDescriptor.StandardScaler,
        TargetTransformDescriptor.MinMaxScaler,
        TargetTransformDescriptor.RobustScaler,
    }

    def __init__(
        self, strategy: TargetTransformDescriptor = TargetTransformDescriptor.NoTransformation
    ) -> None:
        self.strategy = (
            TargetTransformDescriptor(strategy) if isinstance(strategy, str) else strategy
        )
        self._scaler = None

    def fit(self, y: np.ndarray) -> "TargetScaler":
        """
        Fit the scaler on training target values.

        Parameters
        ----------
        y:
            1-D array of training target values.

        Returns
        -------
        TargetScaler
            The fitted scaler (for method chaining).

        Raises
        ------
        ValueError
            If ``strategy`` is ``Log10`` and any value in ``y`` is ≤ 0,
            or if ``strategy`` is ``Log1p`` and any value in ``y`` is ≤ -1.
        """
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

        _SKLEARN_MAP = {
            TargetTransformDescriptor.StandardScaler: StandardScaler,
            TargetTransformDescriptor.MinMaxScaler: MinMaxScaler,
            TargetTransformDescriptor.RobustScaler: RobustScaler,
        }

        y = np.asarray(y, dtype=float).ravel()

        if self.strategy == TargetTransformDescriptor.NoTransformation:
            return self

        if self.strategy in self._SKLEARN_STRATEGIES:
            self._scaler = _SKLEARN_MAP[self.strategy]()
            self._scaler.fit(y.reshape(-1, 1))
            return self

        if self.strategy == TargetTransformDescriptor.Log10:
            if np.any(y <= 0):
                raise ValueError(
                    "TargetScaler with strategy 'log10' requires all training target "
                    f"values to be strictly positive, but found values ≤ 0 in y."
                )
            return self

        if self.strategy == TargetTransformDescriptor.Log1p:
            if np.any(y <= -1):
                raise ValueError(
                    "TargetScaler with strategy 'log1p' requires all training target "
                    f"values to be > -1, but found values ≤ -1 in y."
                )
            return self

        raise ValueError(f"Unsupported TargetTransformDescriptor: '{self.strategy}'.")

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Apply the fitted scaling to ``y``.

        Parameters
        ----------
        y:
            Target values to scale (1-D array or array-like).

        Returns
        -------
        np.ndarray
            Scaled values with the same shape as the input.
        """
        y = np.asarray(y, dtype=float).ravel()

        if self.strategy == TargetTransformDescriptor.NoTransformation:
            return y

        if self.strategy in self._SKLEARN_STRATEGIES:
            return self._scaler.transform(y.reshape(-1, 1)).ravel()

        if self.strategy == TargetTransformDescriptor.Log10:
            return np.log10(y)

        if self.strategy == TargetTransformDescriptor.Log1p:
            return np.log1p(y)

        raise ValueError(f"Unsupported TargetTransformDescriptor: '{self.strategy}'.")

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Reverse the scaling applied by ``transform``.

        Parameters
        ----------
        y:
            Scaled target values to invert (1-D array or array-like).

        Returns
        -------
        np.ndarray
            Values in the original target range.
        """
        y = np.asarray(y, dtype=float).ravel()

        if self.strategy == TargetTransformDescriptor.NoTransformation:
            return y

        if self.strategy in self._SKLEARN_STRATEGIES:
            return self._scaler.inverse_transform(y.reshape(-1, 1)).ravel()

        if self.strategy == TargetTransformDescriptor.Log10:
            return 10.0**y

        if self.strategy == TargetTransformDescriptor.Log1p:
            return np.expm1(y)

        raise ValueError(f"Unsupported TargetTransformDescriptor: '{self.strategy}'.")
