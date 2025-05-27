from typing import Dict

import pandas as pd

from polynet.app.options.data import DataOptions
from polynet.app.options.representation import RepresentationOptions
from polynet.featurizer.descriptor_calculation import calculate_descriptors
from polynet.options.enums import DescriptorMergingMethods


def get_data_index(
    data: pd.DataFrame, id_col: str, smiles_cols: list, weights_col: Dict, target_col: str
):
    return data[[id_col] + smiles_cols + list(weights_col.values()) + [target_col]].copy()


def get_unique_smiles(data: pd.DataFrame, smiles_cols: list) -> list:
    unique = []
    for col in smiles_cols:
        unique.extend(data[col].unique().tolist())
    return unique


def calculate_rdkit_df_dict(unique_smiles, data, smiles_cols, rdkit_descriptors):
    rdkit_desc_dict, columns = calculate_descriptors(
        unique_smiles, descriptors_list=rdkit_descriptors
    )
    descriptors_df = pd.DataFrame.from_dict(rdkit_desc_dict, orient="index", columns=columns)

    rdkit_df_dict = {}
    for col in smiles_cols:
        joined = data.join(descriptors_df, how="left", on=col)
        rdkit_df_dict[col] = joined.drop(columns=data.columns)

    return rdkit_df_dict


def merge_weighted(rdkit_df_dict, data, weights_col, data_index):
    weighted = {
        col: df.multiply(data[weights_col[col]], axis=0) for col, df in rdkit_df_dict.items()
    }
    combined = sum(weighted.values())
    return pd.concat([data_index, combined], axis=1)


def merge_average(rdkit_df_dict, data_index):
    avg = sum(rdkit_df_dict.values()) / len(rdkit_df_dict)
    return pd.concat([data_index, avg], axis=1)


def merge_concatenate(rdkit_df_dict, data_index):
    renamed = {
        key: df.rename(columns={col: f"{key}_{col}" for col in df.columns})
        for key, df in rdkit_df_dict.items()
    }
    combined = pd.concat(renamed.values(), axis=1)
    return pd.concat([data_index, combined], axis=1)


def build_vector_representation(
    representation_opts: RepresentationOptions, data_options: DataOptions, data: pd.DataFrame
) -> pd.DataFrame:
    smiles_cols = data_options.smiles_cols
    id_col = data_options.id_col
    target_col = data_options.target_variable_col
    weights_col = representation_opts.weights_col

    data_index = get_data_index(data, id_col, smiles_cols, weights_col, target_col)
    unique_smiles = get_unique_smiles(data, smiles_cols)

    rdkit_df_dict = {}
    if representation_opts.rdkit_descriptors:
        rdkit_df_dict = calculate_rdkit_df_dict(
            unique_smiles, data, smiles_cols, representation_opts.rdkit_descriptors
        )

    # RDKit-based representation
    if representation_opts.rdkit_independent or representation_opts.mix_rdkit_df_descriptors:
        df_rdkit_weighted = df_rdkit_mean = df_rdkit_concat = None

        if (
            DescriptorMergingMethods.WeightedAverage in representation_opts.smiles_merge_approach
            and weights_col
        ):
            df_rdkit_weighted = merge_weighted(rdkit_df_dict, data, weights_col, data_index)

        if DescriptorMergingMethods.Average in representation_opts.smiles_merge_approach:
            df_rdkit_mean = merge_average(rdkit_df_dict, data_index)

        # TODO: fix case where only one SMILES column is present
        if (
            DescriptorMergingMethods.Concatenate in representation_opts.smiles_merge_approach
            or not representation_opts.smiles_merge_approach
        ):
            df_rdkit_concat = merge_concatenate(rdkit_df_dict, data_index)

    else:
        df_rdkit_weighted = df_rdkit_mean = df_rdkit_concat = None

    # Independent user-provided descriptors
    descriptors_df = (
        data[representation_opts.df_descriptors].copy()
        if (
            representation_opts.df_descriptors_independent
            or representation_opts.mix_rdkit_df_descriptors
        )
        else None
    )

    if representation_opts.mix_rdkit_df_descriptors:
        if df_rdkit_weighted is not None:
            df_rdkit_weighted = pd.concat([df_rdkit_weighted, descriptors_df], axis=1)
        if df_rdkit_mean is not None:
            df_rdkit_mean = pd.concat([df_rdkit_mean, descriptors_df], axis=1)
        if df_rdkit_concat is not None:
            df_rdkit_concat = pd.concat([df_rdkit_concat, descriptors_df], axis=1)

    df_descriptors_df = (
        pd.concat([data_index, descriptors_df], axis=1)
        if representation_opts.df_descriptors_independent
        else None
    )

    # Return dictionary or selected final df depending on use-case
    return {
        "weighted": df_rdkit_weighted,
        "average": df_rdkit_mean,
        "concatenated": df_rdkit_concat,
        "independent_descriptors": df_descriptors_df,
    }
