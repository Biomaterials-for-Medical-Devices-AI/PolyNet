import pandas as pd
from polynet.options.enums import DescriptorMergingMethods
from rdkit.Chem import Descriptors, MolFromSmiles
from polynet.options.enums import MolecularDescriptors
from polynet.featurizer.preprocess import get_data_index
from polynet.utils.chem_utils import PS


def build_vector_representation(
    molecular_descriptors: dict[MolecularDescriptors, str],
    smiles_cols: list[str],
    id_col: str,
    descriptor_merging_approach: DescriptorMergingMethods,
    target_col: str,
    weights_col: dict[str, str],
    data: pd.DataFrame,
    rdkit_independent: bool,
    df_descriptors_independent: bool,
    mix_rdkit_df_descriptors: bool,
) -> pd.DataFrame:

    # TODO: think of a way to not have 3 different bools to know if user wants to merge representations and/or keep them independent

    data_index = get_data_index(data, id_col, smiles_cols, weights_col, target_col)
    unique_smiles = get_unique_smiles(data, smiles_cols)

    rdkit_df_dict = {}
    if MolecularDescriptors.RDKit in molecular_descriptors.keys():
        rdkit_df_dict = calculate_rdkit_df_dict(
            unique_smiles, data, smiles_cols, molecular_descriptors[MolecularDescriptors.RDKit]
        )

    # RDKit-based representation
    if rdkit_independent or mix_rdkit_df_descriptors:
        # df_rdkit_weighted = df_rdkit_mean = df_rdkit_concat = df_rdkit_no_merge = None

        if (
            DescriptorMergingMethods.WeightedAverage == descriptor_merging_approach
        ) and weights_col:
            rdkit_descriptors = merge_weighted(rdkit_df_dict, data, weights_col, data_index)

        elif DescriptorMergingMethods.Average == descriptor_merging_approach:
            rdkit_descriptors = merge_average(rdkit_df_dict, data_index)

        # TODO: fix case where only one SMILES column is present
        elif DescriptorMergingMethods.Concatenate == descriptor_merging_approach:
            rdkit_descriptors = merge_concatenate(rdkit_df_dict, data_index)

        elif DescriptorMergingMethods.NoMerging == descriptor_merging_approach:
            rdkit_descriptors = single_smiles(rdkit_df_dict, data_index)
        else:
            raise Exception("Invalid merging method selected.")

    else:
        rdkit_descriptors = None

    # Independent user-provided descriptors
    descriptors_df = (
        data[molecular_descriptors[MolecularDescriptors.DataFrame]].copy()
        if (df_descriptors_independent or mix_rdkit_df_descriptors)
        else None
    )

    if mix_rdkit_df_descriptors:
        rdkit_df_descriptors = pd.concat([rdkit_descriptors, descriptors_df], axis=1)
    else:
        rdkit_df_descriptors = None

    df_descriptors_df = (
        pd.concat([data_index, descriptors_df], axis=1) if df_descriptors_independent else None
    )

    if not rdkit_independent:
        rdkit_descriptors = None

    # polyBERT_df_dict = {}
    # polyBERT_descriptors = None
    # if representation_opts.polybert_fp:
    #     polyBERT_fps = get_polyBERT_fps(psmiles=unique_smiles)
    #     polyBERT_df_dict = calculate_polybert_df_dict(polyBERT_fps, data, smiles_cols)

    #     if DescriptorMergingMethods.WeightedAverage == smiles_merge_approach and weights_col:
    #         polyBERT_descriptors = merge_weighted(polyBERT_df_dict, data, weights_col, data_index)

    #     elif DescriptorMergingMethods.Average == smiles_merge_approach:
    #         polyBERT_descriptors = merge_average(polyBERT_df_dict, data_index)

    #     elif DescriptorMergingMethods.Concatenate == descriptpr_merging_approach:
    #         polyBERT_descriptors = merge_concatenate(polyBERT_df_dict, data_index)

    #     elif DescriptorMergingMethods.NoMerging == descriptpr_merging_approach:
    #         polyBERT_descriptors = single_smiles(polyBERT_df_dict, data_index)

    # Return dictionary or selected final df depending on use-case
    return {
        MolecularDescriptors.RDKit: rdkit_descriptors,
        MolecularDescriptors.DataFrame: df_descriptors_df,
        MolecularDescriptors.RDKit_DataFrame: rdkit_df_descriptors,
        # "polyBERT": polyBERT_descriptors,
    }


def calculate_rdkit_df_dict(
    unique_smiles: list[str],
    data: pd.DataFrame,
    smiles_cols: list[str],
    rdkit_descriptors: list[str],
):
    """
    Calculate RDKit molecular descriptors for a set of unique SMILES strings and merge
    the resulting descriptor values into the provided DataFrame for each SMILES column.

    This function computes RDKit descriptors once for the unique SMILES to avoid redundant
    calculations, constructs a descriptor DataFrame, and then merges it with the input data
    on each SMILES column provided. The resulting DataFrames contain only the descriptor
    columns, aligned with the original data by SMILES.

    Args:
        unique_smiles (list[str]): List of unique SMILES strings to compute descriptors for.
        data (pd.DataFrame): Original dataset containing one or more SMILES columns.
        smiles_cols (list[str]): List of column names in `data` that contain SMILES strings.
        rdkit_descriptors (list[str]): List of RDKit descriptor names to calculate.

    Returns:
        dict[str, pd.DataFrame]:
            A dictionary mapping each SMILES column name to a DataFrame containing
            the calculated RDKit descriptors for that column.

            Example:
                {
                    "reactant_smiles": pd.DataFrame(...),
                    "product_smiles": pd.DataFrame(...),
                }

    Notes:
        - The function uses `calculate_descriptors()` internally to compute descriptor values.
        - Each resulting DataFrame contains descriptor columns only (the original
          columns from `data` are dropped after joining).
        - Missing or invalid SMILES will result in skipped entries.
    """
    rdkit_desc_dict, columns = calculate_descriptors(
        smiles_list=unique_smiles, descriptors_list=rdkit_descriptors
    )
    descriptors_df = pd.DataFrame.from_dict(rdkit_desc_dict, orient="index", columns=columns)

    rdkit_df_dict = {}
    for col in smiles_cols:
        joined = data.join(descriptors_df, how="left", on=col)
        rdkit_df_dict[col] = joined.drop(columns=data.columns)

    return rdkit_df_dict


def calculate_descriptors(
    smiles_list: list[str], descriptors_list: list[str] | str = "all"
) -> tuple[dict[str, list[float]], list[str]]:
    """
    Calculate molecular descriptors for a list of SMILES strings.

    Args:
        smiles_list (list[str]): List of SMILES strings.
        descriptors_list (list[str] | str): List of descriptor names to calculate or "all" to calculate all.

    Returns:
        tuple: A dictionary mapping each SMILES string to its descriptors,
               and a list of the descriptor names used.
    """
    all_descriptor_funcs = dict(Descriptors.descList)

    # Validate and select descriptor functions
    if descriptors_list == "all":
        selected_descriptors = all_descriptor_funcs
    else:
        if not isinstance(descriptors_list, list):
            raise ValueError("descriptors_list must be a list of strings or 'all'.")

        selected_descriptors = {
            name: all_descriptor_funcs[name]
            for name in descriptors_list
            if name in all_descriptor_funcs
        }

        # Warn if any descriptors were not found
        missing = set(descriptors_list) - set(selected_descriptors)
        if missing:
            print(
                f"Warning: The following descriptors were not found and will be skipped: {missing}"
            )

    descriptors = {}

    for smiles in smiles_list:
        mol = MolFromSmiles(smiles)

        if mol is None:
            print(f"Error processing molecule: {smiles}")
            continue

        descriptors[smiles] = [func(mol) for func in selected_descriptors.values()]

    return descriptors, list(selected_descriptors.keys())


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


def get_unique_smiles(data: pd.DataFrame, smiles_cols: list) -> list:
    unique = []
    for col in smiles_cols:
        unique.extend(data[col].unique().tolist())
    return unique
