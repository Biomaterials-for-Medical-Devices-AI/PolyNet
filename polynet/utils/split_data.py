import pandas as pd
from sklearn.model_selection import train_test_split

from polynet.options.enums import EvaluationMetrics, ProblemTypes, SplitMethods, SplitTypes
from polynet.utils.data_preprocessing import class_balancer


def split_data(data, test_size, random_state, stratify=None):
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


def get_data_split_indices(
    data: pd.DataFrame,
    split_type: SplitTypes,
    n_bootstrap_iterations: int,
    val_ratio: float,
    test_ratio: float,
    target_variable_col: str,
    split_method: SplitMethods,
    train_set_balance: float,
    random_seed: int,
):

    if split_type == SplitTypes.TrainValTest:

        train_data_idxs, val_data_idxs, test_data_idxs = [], [], []

        for i in range(n_bootstrap_iterations):
            # Initial train-test split
            train_data, test_data = split_data(
                data=data,
                test_size=test_ratio,
                stratify=(
                    data[target_variable_col] if split_method == SplitMethods.Stratified else None
                ),
                random_state=random_seed + i,
            )

            # Optional class balancing on training set
            if train_set_balance:
                train_data = class_balancer(
                    data=train_data,
                    target=target_variable_col,
                    desired_class_proportion=train_set_balance,
                    random_state=random_seed + i,
                )

            # Further split train into train/validation
            train_data, val_data = split_data(
                data=train_data,
                test_size=val_ratio,
                stratify=(
                    train_data[target_variable_col]
                    if split_method == SplitMethods.Stratified
                    else None
                ),
                random_state=random_seed + i,
            )

            train_data_idxs.append(train_data.index.tolist())
            val_data_idxs.append(val_data.index.tolist())
            test_data_idxs.append(test_data.index.tolist())

    return train_data_idxs, val_data_idxs, test_data_idxs
