from pathlib import Path

import pandas as pd
from polynet.options.enums import Results


def create_directory(path: Path):
    """Create a directory at the specified path. If intermediate directories
    don't already exist, create them. If the path already exists, no action
    is taken.

    Args:
        path (Path): The path the directory to create.
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def keep_only_numerical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DataFrame columns to numeric where possible.
    Drops any column that cannot be converted.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing only numerical columns.
    """
    numeric_df = pd.DataFrame()

    for col in df.columns:
        try:
            numeric_col = pd.to_numeric(df[col], errors="raise")
            numeric_df[col] = numeric_col
        except (ValueError, TypeError):
            # Column could not be converted to numeric; skip it
            continue

    return numeric_df


def check_column_is_numeric(df, column_name: str) -> bool:
    """Check if a column in a DataFrame is numeric.

    Args:
        df (pd.DataFrame): The DataFrame to check.
        column_name (str): The name of the column to check.

    Returns:
        bool: True if the column is numeric, False otherwise.
    """
    return pd.api.types.is_numeric_dtype(df[column_name]) if column_name in df.columns else False


def save_data(data_path: Path, data: pd.DataFrame):
    """Save data to either a '.csv' or '.xlsx' file.

    Args:
        data_path (Path): The path to save the data to.
        data (pd.DataFrame): The data to save.
        logger (Logger): The logger.

    Raises:
        ValueError: The data file wasn't a '.csv' or '.xlsx' file.
    """
    if data_path.suffix == ".csv":
        try:
            data.to_csv(data_path, index=False)
        except Exception as e:
            raise
    elif data_path.suffix == ".xlsx":
        try:
            data.to_excel(data_path, index=False)
        except Exception as e:
            raise
    else:
        raise ValueError("data_path must be to a '.csv' or '.xlsx' file")


def merge_model_predictions(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    # Start with the first dataframe as base
    base_df = dfs[0].copy()

    # Keep track of new prediction columns
    new_pred_cols = []

    for df in dfs[1:]:
        # Identify the new prediction column
        pred_col = [col for col in df.columns if col not in base_df.columns][0]
        new_pred_cols.append(pred_col)

        # Merge prediction column by index
        base_df = base_df.merge(df[[Results.Index.value, pred_col]], on=Results.Index.value)

    # Reorder columns: all existing base columns first, then all prediction columns
    existing_cols = [col for col in base_df.columns if col not in new_pred_cols]
    final_columns = existing_cols + new_pred_cols

    return base_df[final_columns]
