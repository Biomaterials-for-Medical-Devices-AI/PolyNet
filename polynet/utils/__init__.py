from pathlib import Path
import re

import numpy as np
import pandas as pd

from polynet.config.column_names import get_score_column_name


def prepare_probs_df(probs: np.ndarray, target_variable_name: str = None, model_name: str = None):
    """
    Convert probability predictions into a DataFrame.

    - For binary classification (2 classes), include only the second class (index 1).
    - For multi-class classification (3+ classes), include a column per class.

    Args:
        probs (np.ndarray): Array of shape (n_samples, n_classes)
        target_variable_name (str): Name of the target variable
        model_name (str): Name of the model

    Returns:
        pd.DataFrame: A DataFrame with appropriately named probability columns
    """
    n_classes = probs.shape[1] if probs.ndim > 1 else 1
    probs_df = pd.DataFrame()

    if n_classes == 2:
        col_name = get_score_column_name(
            target_variable_name=target_variable_name, model_name=model_name
        )
        # Binary classification: only use the second class (probability of class 1)
        probs_df[f"{col_name}"] = probs[:, 1]
    else:
        # Multi-class classification: create one column per class
        for i in range(n_classes):
            col_name = get_score_column_name(
                target_variable_name=target_variable_name, model_name=model_name, class_num=i
            )
            probs_df[f"{col_name} {i}"] = probs[:, i]

    return probs_df


def create_directory(path: Path):
    """Create a directory at the specified path, including any intermediate directories.

    If the path already exists, no action is taken.

    Args:
        path (Path): The path of the directory to create.
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def save_data(data_path: Path, data: pd.DataFrame):
    """Save a DataFrame to a CSV or Excel file.

    Args:
        data_path (Path): The path to save the data to (.csv or .xlsx).
        data (pd.DataFrame): The data to save.

    Raises:
        ValueError: If the file extension is not .csv or .xlsx.
    """
    if data_path.suffix == ".csv":
        data.to_csv(data_path, index=False)
    elif data_path.suffix == ".xlsx":
        data.to_excel(data_path, index=False)
    else:
        raise ValueError("data_path must be to a '.csv' or '.xlsx' file")


def filter_dataset_by_ids(dataset, ids):
    """Filter a graph dataset to only include samples whose idx is in ids.

    Args:
        dataset: Iterable of graph data objects with an ``idx`` attribute.
        ids: Collection of ids to keep.

    Returns:
        list: Filtered list of data objects.
    """
    return [data for data in dataset if data.idx in ids]


def extract_number(filename):
    """Extract the trailing integer from a model filename like ``model_3.pt``.

    Args:
        filename (str): The filename to parse.

    Returns:
        int: The extracted number.

    Raises:
        ValueError: If no trailing number is found.
    """
    match = re.search(r"_(\d+)\.pt$", filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"No number found in filename: {filename}")
