from typing import Dict

import pandas as pd

from polynet.app.options.data import DataOptions
from polynet.app.options.representation import RepresentationOptions
from polynet.featurizer.descriptor_calculation import calculate_descriptors
from polynet.options.enums import DescriptorMergingMethods
from polynet.utils.chem_utils import PS


def get_data_index(
    data: pd.DataFrame, id_col: str, smiles_cols: list, weights_col: Dict, target_col: str
):
    idx_cols = [id_col] + smiles_cols + list(weights_col.values())
    if target_col in data.columns:
        idx_cols += [target_col]
    return data[idx_cols].copy()


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
        col: df.multiply(data[weights_col[col]], axis=0) / 100 for col, df in rdkit_df_dict.items()
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


def single_smiles(rdkit_df_dict, data_index):
    # No merging, just return the data index
    return pd.concat([data_index] + list(rdkit_df_dict.values()), axis=1)


def get_polyBERT_fps(psmiles: list):

    pb_fps = {}

    for psmile in psmiles:
        fingerprint = PS(psmile).fingerprint("polyBERT")
        pb_fps[psmile] = fingerprint

    return pb_fps


def calculate_polybert_df_dict(polyBERT_fps: dict, data: pd.DataFrame, smiles_cols: list):
    # Create a single DataFrame for all polyBERT fingerprints
    columns = [f"polyBERT_{i}" for i in range(len(next(iter(polyBERT_fps.values()))))]
    polybert_df = pd.DataFrame.from_dict(polyBERT_fps, orient="index", columns=columns)

    polybert_df_dict = {}
    for col in smiles_cols:
        joined = data.join(polybert_df, how="left", on=col)
        polybert_df_dict[col] = joined.drop(columns=data.columns)

    return polybert_df_dict


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
        # df_rdkit_weighted = df_rdkit_mean = df_rdkit_concat = df_rdkit_no_merge = None

        if (
            DescriptorMergingMethods.WeightedAverage == representation_opts.smiles_merge_approach
        ) and weights_col:
            rdkit_descriptors = merge_weighted(rdkit_df_dict, data, weights_col, data_index)

        elif DescriptorMergingMethods.Average == representation_opts.smiles_merge_approach:
            rdkit_descriptors = merge_average(rdkit_df_dict, data_index)

        # TODO: fix case where only one SMILES column is present
        elif DescriptorMergingMethods.Concatenate == representation_opts.smiles_merge_approach:
            rdkit_descriptors = merge_concatenate(rdkit_df_dict, data_index)

        elif DescriptorMergingMethods.NoMerging == representation_opts.smiles_merge_approach:
            rdkit_descriptors = single_smiles(rdkit_df_dict, data_index)

    else:
        rdkit_descriptors = None

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
        rdkit_df_descriptors = pd.concat([rdkit_descriptors, descriptors_df], axis=1)
    else:
        rdkit_df_descriptors = None

    df_descriptors_df = (
        pd.concat([data_index, descriptors_df], axis=1)
        if representation_opts.df_descriptors_independent
        else None
    )

    if not representation_opts.rdkit_independent:
        rdkit_descriptors = None

    polyBERT_df_dict = {}
    polyBERT_descriptors = None
    if representation_opts.polybert_fp:
        polyBERT_fps = get_polyBERT_fps(psmiles=unique_smiles)
        polyBERT_df_dict = calculate_polybert_df_dict(polyBERT_fps, data, smiles_cols)

        if (
            DescriptorMergingMethods.WeightedAverage == representation_opts.smiles_merge_approach
            and weights_col
        ):
            polyBERT_descriptors = merge_weighted(polyBERT_df_dict, data, weights_col, data_index)

        elif DescriptorMergingMethods.Average == representation_opts.smiles_merge_approach:
            polyBERT_descriptors = merge_average(polyBERT_df_dict, data_index)

        elif DescriptorMergingMethods.Concatenate == representation_opts.smiles_merge_approach:
            polyBERT_descriptors = merge_concatenate(polyBERT_df_dict, data_index)

        elif DescriptorMergingMethods.NoMerging == representation_opts.smiles_merge_approach:
            polyBERT_descriptors = single_smiles(polyBERT_df_dict, data_index)

    # Return dictionary or selected final df depending on use-case
    return {
        "RDKit": rdkit_descriptors,
        "DF": df_descriptors_df,
        "RDKit_DF": rdkit_df_descriptors,
        "polyBERT": polyBERT_descriptors,
    }
