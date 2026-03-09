import pandas as pd


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
