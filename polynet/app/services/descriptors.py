from polynet.featurizer.descriptor_calculation import calculate_descriptors
from polynet.app.options.representation import RepresentationOptions
from polynet.app.options.data import DataOptions
import pandas as pd
from polynet.options.enums import DescriptorMergingMethods


def build_vector_representation(
    representation_opts: RepresentationOptions, data_options: DataOptions, data: pd.DataFrame
) -> pd.DataFrame:

    rdkit_descriptors = representation_opts.rdkit_descriptors
    df_descriptors = representation_opts.df_descriptors

    smiles_cols = data_options.smiles_cols
    id_col = data_options.id_col
    target_variable_col = data_options.target_variable_col
    weights_col = representation_opts.weights_col

    data_index = data[
        [id_col] + smiles_cols + list(weights_col.values()) + [target_variable_col]
    ].copy()

    unique_smiles = []
    for col in smiles_cols:
        unique_smiles.extend(data[col].unique().tolist())

    if rdkit_descriptors:
        rdkit_df_dict = {}
        rdkit_descriptors, columns = calculate_descriptors(
            smiles_list=unique_smiles, descriptors_list=rdkit_descriptors
        )
        descriptors_df = pd.DataFrame.from_dict(rdkit_descriptors, orient="index", columns=columns)
        for col in smiles_cols:
            rdkit_df_dict[col] = data.join(descriptors_df, how="left", on=col).drop(
                columns=data.columns
            )

    if representation_opts.rdkit_independent or representation_opts.mix_rdkit_df_descriptors:
        if (
            DescriptorMergingMethods.WeightedAverage in representation_opts.smiles_merge_approach
            and weights_col
        ):
            rdkit_df_dict_weighted = rdkit_df_dict.copy()
            for col in rdkit_df_dict_weighted:
                weight_factor_col = weights_col[col]
                rdkit_df_dict_weighted[col] = rdkit_df_dict_weighted[col].multiply(
                    data[weight_factor_col], axis=0
                )

            rdkit_descriptors_df_weighted = sum(rdkit_df_dict_weighted.values())
            rdkit_descriptors_df_weighted = pd.concat(
                [data_index, rdkit_descriptors_df_weighted], axis=1
            )

        else:
            rdkit_descriptors_df_weighted = None

        if DescriptorMergingMethods.Average in representation_opts.smiles_merge_approach:
            rdkit_descriptors_df_mean = sum(rdkit_df_dict.values()) / len(rdkit_df_dict)
            rdkit_descriptors_df_mean = pd.concat([data_index, rdkit_descriptors_df_mean], axis=1)
        else:
            rdkit_descriptors_df_mean = None

        if DescriptorMergingMethods.Concatenate in representation_opts.smiles_merge_approach:
            rdkit_df_dict_concat = rdkit_df_dict.copy()
            for key, value in rdkit_df_dict_concat.items():
                rdkit_df_dict_concat[key] = rdkit_df_dict_concat[key].rename(
                    columns={col: f"{key}_{col}" for col in value.columns}
                )
            rdkit_descriptors_df_concat = pd.concat(rdkit_df_dict_concat.values(), axis=1)
            rdkit_descriptors_df_concat = pd.concat(
                [data_index, rdkit_descriptors_df_concat], axis=1
            )
        else:
            rdkit_descriptors_df_concat = None

    if (
        representation_opts.df_descriptors_independent
        or representation_opts.mix_rdkit_df_descriptors
    ):
        descriptors_df = data[df_descriptors].copy()
    else:
        descriptors_df = None

    if representation_opts.mix_rdkit_df_descriptors:
        if rdkit_descriptors_df_weighted is not None:
            rdkit_descriptors_df_weighted = pd.concat(
                [rdkit_descriptors_df_weighted, descriptors_df], axis=1
            )
        if rdkit_descriptors_df_mean is not None:
            rdkit_descriptors_df_mean = pd.concat(
                [rdkit_descriptors_df_mean, descriptors_df], axis=1
            )
        if rdkit_descriptors_df_concat is not None:
            rdkit_descriptors_df_concat = pd.concat(
                [rdkit_descriptors_df_concat, descriptors_df], axis=1
            )

    if representation_opts.df_descriptors_independent:
        df_descriptors_df = pd.concat([data_index, descriptors_df], axis=1)

    else:
        df_descriptors_df = None
