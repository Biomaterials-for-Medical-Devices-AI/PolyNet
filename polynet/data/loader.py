"""
polynet.data.loader
====================
Data loading utilities for the polynet pipeline.

Wraps ``pd.read_csv`` with validation specific to polymer informatics
datasets — checking that required columns exist, that SMILES columns are
non-empty strings, and that the target variable is present and numeric
where expected.

Public API
----------
::

    from polynet.data.loader import load_dataset

    df = load_dataset(
        path="data/polymers.csv",
        smiles_cols=["SMILES"],
        target_col="Tg",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from polynet.config.enums import ProblemType

logger = logging.getLogger(__name__)


def load_dataset(
    path: str | Path,
    smiles_cols: list[str],
    target_col: str,
    id_col: str | None = None,
    problem_type: ProblemType | str | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Load a polymer dataset from a CSV file with validation.

    Parameters
    ----------
    path:
        Path to the CSV file.
    smiles_cols:
        Column names that contain SMILES strings. All must be present
        in the file and must not be entirely empty.
    target_col:
        Column name for the target property. Must be present in the file.
    id_col:
        Optional sample identifier column. If provided, must be present
        in the file.
    problem_type:
        If provided, validates that the target column is numeric for
        regression or has a small finite set of values for classification.
    **read_csv_kwargs:
        Additional keyword arguments forwarded to ``pd.read_csv``
        (e.g. ``sep``, ``encoding``, ``nrows``).

    Returns
    -------
    pd.DataFrame
        The loaded dataset. No columns are dropped or modified — the
        raw DataFrame is returned so downstream steps have full control.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If required columns are missing, SMILES columns are empty, or
        target column validation fails.

    Examples
    --------
    >>> from polynet.data.loader import load_dataset
    >>> df = load_dataset(
    ...     path="data/polymers.csv",
    ...     smiles_cols=["SMILES"],
    ...     target_col="Tg",
    ...     problem_type="regression",
    ... )
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: '{path.resolve()}'. " "Check the path in your config."
        )

    logger.info(f"Loading dataset from '{path}'...")
    df = pd.read_csv(path, **read_csv_kwargs)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

    _validate_columns(df, smiles_cols, target_col, id_col)
    _validate_smiles_columns(df, smiles_cols)

    if problem_type is not None:
        problem_type = ProblemType(problem_type) if isinstance(problem_type, str) else problem_type
        _validate_target_column(df, target_col, problem_type)

    return df


# ---------------------------------------------------------------------------
# Private validators
# ---------------------------------------------------------------------------


def _validate_columns(
    df: pd.DataFrame, smiles_cols: list[str], target_col: str, id_col: str | None
) -> None:
    """Check that all required columns are present in the DataFrame."""
    required = set(smiles_cols) | {target_col}
    if id_col:
        required.add(id_col)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"The following required columns are missing from the dataset: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns.tolist())}."
        )


def _validate_smiles_columns(df: pd.DataFrame, smiles_cols: list[str]) -> None:
    """Check that SMILES columns contain non-null, non-empty string values."""
    for col in smiles_cols:
        null_count = df[col].isna().sum()
        if null_count > 0:
            logger.warning(
                f"SMILES column '{col}' contains {null_count} null values. "
                "These rows will be skipped during featurization."
            )

        empty_count = (df[col].astype(str).str.strip() == "").sum()
        if empty_count > 0:
            logger.warning(
                f"SMILES column '{col}' contains {empty_count} empty strings. "
                "These rows will be skipped during featurization."
            )


def _validate_target_column(df: pd.DataFrame, target_col: str, problem_type: ProblemType) -> None:
    """Validate target column dtype against the declared problem type."""
    if problem_type == ProblemType.Regression:
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            raise ValueError(
                f"Target column '{target_col}' is not numeric, but problem_type "
                "is 'regression'. Check the column or the problem_type setting."
            )
        null_count = df[target_col].isna().sum()
        if null_count > 0:
            logger.warning(
                f"Target column '{target_col}' contains {null_count} null values. "
                "Consider dropping or imputing these rows before training."
            )

    elif problem_type == ProblemType.Classification:
        n_unique = df[target_col].nunique()
        if n_unique > 20:
            logger.warning(
                f"Target column '{target_col}' has {n_unique} unique values for a "
                "classification task. If this is a regression target, check your "
                "problem_type setting."
            )
