import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from polynet.options.enums import TransformDescriptors


def transform_dependent_variables(
    fit_data: pd.DataFrame, transform_data: pd.DataFrame, transform_type: TransformDescriptors
):
    """
    Transform the dependent variable based on the specified transformation type.
    """
    if transform_type == TransformDescriptors.StandardScaler:

        scaler = StandardScaler()
        scaler.fit(fit_data)
        transform_data = scaler.transform(transform_data)

    elif transform_type == TransformDescriptors.MinMaxScaler:

        scaler = MinMaxScaler()
        scaler.fit(fit_data)
        transform_data = scaler.transform(transform_data)

    else:
        raise ValueError(f"Unsupported transformation type: {transform_type}")

    return transform_data, scaler


def sanitise_df(df: pd.DataFrame, descriptors: list, target_variable_col: str):

    clean_df = df.copy()

    clean_df = clean_df[descriptors + [target_variable_col]].dropna(axis=1)

    return clean_df


def get_data_index(
    data: pd.DataFrame, id_col: str, smiles_cols: list, weights_col: dict, target_col: str
):
    idx_cols = [id_col] + smiles_cols + list(weights_col.values())
    if target_col in data.columns:
        idx_cols += [target_col]
    return data[idx_cols].copy()
