import numpy as np
import pandas as pd

from polynet.options.col_names import get_score_column_name
import matplotlib.pyplot as plt


def save_plot(fig, path, dpi=300):
    """
    Saves the plot to the specified path.

    Args:
        fig: The figure to save.
        path (str): The path where the plot will be saved.
    """
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    print(f"Plot saved to {path}")
    plt.close()


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
