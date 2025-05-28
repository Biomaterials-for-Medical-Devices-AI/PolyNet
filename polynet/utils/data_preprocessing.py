from collections import Counter
import random

import pandas as pd


def print_class_balance(data: pd.DataFrame, target: str):
    """
    Prints the class balance of the target variable in the
    provided dataset.

    Args:
        data (pd.DataFrame): The dataset containing the target variable.
        target (str): The name of the target column.
    """

    class_counts = Counter(data[target])

    minority_class, majority_class = sorted(class_counts, key=lambda x: class_counts[x])

    # Compute new class distribution
    new_counts = Counter(data[target])
    new_proportion = new_counts[minority_class] / sum(new_counts.values())

    print(
        f"CLASS BALANCE\n[{minority_class}]: {new_counts[minority_class]}  "
        f"[{majority_class}]: {new_counts[majority_class]}  "
        f"({new_proportion:.2f}/{1-new_proportion:.2f})"
    )


def class_balancer(
    desired_class_proportion: float, data: pd.DataFrame, target: str, random_state: int = 1
):
    """
    Balances the dataset by undersampling the majority class to achieve the desired class proportion.

    Args:
        desired_class_proportion (float): Desired proportion of the minority class (e.g., 0.6 means 60% of the minority class).
        data (pd.DataFrame): The dataset containing the target variable.
        target (str): The name of the target column.

    Returns:
        pd.DataFrame: A balanced dataset with adjusted class proportions.
    """

    # Validate inputs
    if data is None or target is None:
        raise ValueError("Data and target must be provided.")

    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in the dataset.")

    if not (0 < desired_class_proportion < 1):
        raise ValueError("desired_class_proportion must be between 0 and 1.")

    # Count occurrences of each class
    class_counts = Counter(data[target])

    if len(class_counts) != 2:
        raise ValueError("The target variable must be binary (contain only two unique values).")

    # Identify minority and majority class
    minority_class, majority_class = sorted(class_counts, key=lambda x: class_counts[x])

    # Compute number of samples to remove
    minority_size = class_counts[minority_class]
    majority_size = class_counts[majority_class]

    samples_to_remove = majority_size - int(
        minority_size * desired_class_proportion / (1 - desired_class_proportion)
    )

    if samples_to_remove <= 0:
        print(
            f"No balancing needed. The dataset already meets the desired proportion ({desired_class_proportion})."
        )
        return data

    print(f"Samples to remove: {samples_to_remove}")

    # Randomly select rows to drop
    drop_indices = (
        data[data[target] == majority_class]
        .sample(n=samples_to_remove, random_state=random_state)
        .index
    )
    data_balanced = data.drop(index=drop_indices)

    print_class_balance(data_balanced, target)

    return data_balanced
