from sklearn.model_selection import train_test_split


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
