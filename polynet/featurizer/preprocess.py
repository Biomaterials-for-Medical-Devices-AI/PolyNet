import pandas as pd


def sanitise_df(df: pd.DataFrame, descriptors: list, target_variable_col: str):

    clean_df = df.copy()

    clean_df = clean_df[descriptors + [target_variable_col]].dropna(axis=1)

    return clean_df
