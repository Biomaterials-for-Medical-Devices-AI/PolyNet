from sklearn.model_selection import train_test_split
import streamlit as st
from torch import load, save


def split_data(data, test_size=0.2, random_state=1, stratify=None):
    """
    Splits the data into training and testing sets.

    Args:
        data (pd.DataFrame): The input data.
        target_col (str): The name of the target column.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
        X_train, X_test, y_train, y_test: The training and testing sets.
    """

    return train_test_split(data, test_size=test_size, random_state=random_state, stratify=stratify)


def save_gnn_model(model, path):
    """
    Saves the GNN model to the specified path.

    Args:
        model: The GNN model to save.
        path (str): The path where the model will be saved.
    """

    save(model, path)


def load_gnn_model(path):
    """
    Loads the GNN model from the specified path.

    Args:
        path (str): The path from which to load the model.

    Returns:
        The loaded GNN model.
    """
    return load(path, weights_only=False)


def save_plot(fig, path):
    """
    Saves the plot to the specified path.

    Args:
        fig: The figure to save.
        path (str): The path where the plot will be saved.
    """
    fig.savefig(path)
    print(f"Plot saved to {path}")
