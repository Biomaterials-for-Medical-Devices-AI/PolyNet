"""
polynet.featurizer.descriptors
================================
Molecular descriptor calculation and merging for polymer datasets.

Supports three descriptor sources:

1. **RDKit** — computed from SMILES strings using the RDKit library.
2. **DataFrame** — user-provided descriptors loaded from an external file.
3. **PolyBERT** — latent-space fingerprints from the PolyBERT model
   (requires PSMILES notation).

For multi-monomer polymers (multiple SMILES columns), per-monomer
descriptors are merged into a single polymer-level representation using
one of the strategies defined in ``DescriptorMergingMethod``.

Public API
----------
::

    from polynet.featurizer.descriptors import build_vector_representation
    from polynet.config.enums import MolecularDescriptor, DescriptorMergingMethod

    result = build_vector_representation(
        data=df,
        molecular_descriptors={MolecularDescriptor.RDKit: ["MolWt", "TPSA"]},
        smiles_cols=["smiles_monomer_1", "smiles_monomer_2"],
        id_col="polymer_id",
        target_col="Tg",
        merging_approach=DescriptorMergingMethod.WeightedAverage,
        weights_col={"smiles_monomer_1": "ratio_1", "smiles_monomer_2": "ratio_2"},
        rdkit_independent=True,
        df_descriptors_independent=False,
        mix_rdkit_df_descriptors=False,
    )
"""

from __future__ import annotations

import logging
import warnings

import pandas as pd
from rdkit.Chem import Descriptors, MolFromSmiles

from polynet.config.enums import DescriptorMergingMethod, MolecularDescriptor
from polynet.data.preprocessing import get_data_index

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_vector_representation(
    data: pd.DataFrame,
    molecular_descriptors: dict[MolecularDescriptor, list[str]],
    smiles_cols: list[str],
    id_col: str | None,
    target_col: str,
    merging_approach: DescriptorMergingMethod | str,
    weights_col: dict[str, str] | None = None,
    rdkit_independent: bool = True,
    df_descriptors_independent: bool = False,
    mix_rdkit_df_descriptors: bool = False,
) -> dict[MolecularDescriptor, pd.DataFrame | None]:
    """
    Build fixed-length vector representations for a polymer dataset.

    Returns a dict keyed by ``MolecularDescriptor`` whose values are
    DataFrames ready for traditional ML training, or ``None`` if that
    representation was not requested.

    Parameters
    ----------
    data:
        Full dataset DataFrame including SMILES, target, and any
        weight/descriptor columns.
    molecular_descriptors:
        Mapping from descriptor type to list of descriptor names.
        Pass an empty list to skip a descriptor type.
    smiles_cols:
        Column names containing SMILES strings.
    id_col:
        Optional sample identifier column.
    target_col:
        Target property column name.
    merging_approach:
        How to merge per-monomer descriptors into a single polymer vector.
    weights_col:
        Mapping from SMILES column to weight fraction column.
        Required when ``merging_approach`` is ``WeightedAverage``.
    rdkit_independent:
        If True, return RDKit descriptors as a standalone representation.
    df_descriptors_independent:
        If True, return DataFrame descriptors as a standalone representation.
    mix_rdkit_df_descriptors:
        If True, concatenate RDKit and DataFrame descriptors.

    Returns
    -------
    dict[MolecularDescriptor, pd.DataFrame | None]
        Keys: ``RDKit``, ``DataFrame``, ``RDKit_DataFrame``.
        Values are DataFrames or ``None`` if not computed.

    Raises
    ------
    ValueError
        If ``WeightedAverage`` merging is requested without ``weights_col``,
        or if an unsupported merging method is provided.
    """
    merging_approach = (
        DescriptorMergingMethod(merging_approach)
        if isinstance(merging_approach, str)
        else merging_approach
    )

    if merging_approach == DescriptorMergingMethod.WeightedAverage and not weights_col:
        raise ValueError(
            "merging_approach is 'weighted_average' but weights_col is not provided. "
            "Supply a mapping from SMILES column to weight fraction column."
        )

    data_index = get_data_index(data, id_col, smiles_cols, weights_col, target_col)
    unique_smiles = _get_unique_smiles(data, smiles_cols)

    # --- RDKit descriptors ---
    rdkit_descriptors_df: pd.DataFrame | None = None
    rdkit_desc_list = molecular_descriptors.get(MolecularDescriptor.RDKit, [])

    if rdkit_desc_list and (rdkit_independent or mix_rdkit_df_descriptors):
        rdkit_df_dict = _calculate_rdkit_df_dict(unique_smiles, data, smiles_cols, rdkit_desc_list)
        rdkit_descriptors_df = _merge(
            rdkit_df_dict, data, weights_col, data_index, merging_approach
        )

    if not rdkit_independent:
        rdkit_descriptors_df = None

    # --- DataFrame descriptors ---
    df_desc_cols = molecular_descriptors.get(MolecularDescriptor.DataFrame, [])
    df_descriptors_df: pd.DataFrame | None = None

    if df_desc_cols and df_descriptors_independent:
        descriptors_df = data[df_desc_cols].copy()
        df_descriptors_df = pd.concat([data_index, descriptors_df], axis=1)

    # --- Mixed RDKit + DataFrame ---
    mixed_df: pd.DataFrame | None = None
    if mix_rdkit_df_descriptors and rdkit_descriptors_df is not None and df_desc_cols:
        descriptors_df = data[df_desc_cols].copy()
        mixed_df = pd.concat([rdkit_descriptors_df, descriptors_df], axis=1)

    # --- PolyBERT ---
    if molecular_descriptors.get(MolecularDescriptor.PolyBERT):
        warnings.warn(
            "PolyBERT fingerprint computation is not yet fully integrated. "
            "The PolyBERT representation will be skipped.",
            UserWarning,
            stacklevel=2,
        )

    return {
        MolecularDescriptor.RDKit: rdkit_descriptors_df,
        MolecularDescriptor.DataFrame: df_descriptors_df,
        MolecularDescriptor.RDKit_DataFrame: mixed_df,
    }


# ---------------------------------------------------------------------------
# RDKit descriptor computation
# ---------------------------------------------------------------------------


def calculate_descriptors(
    smiles_list: list[str], descriptors_list: list[str] | str = "all"
) -> tuple[dict[str, list[float]], list[str]]:
    """
    Calculate RDKit molecular descriptors for a list of SMILES strings.

    Parameters
    ----------
    smiles_list:
        SMILES strings to compute descriptors for.
    descriptors_list:
        List of RDKit descriptor names to compute, or ``"all"`` to
        compute all available descriptors.

    Returns
    -------
    tuple[dict[str, list[float]], list[str]]
        A dict mapping each SMILES to its descriptor values, and the
        list of descriptor names in the same order.

    Raises
    ------
    ValueError
        If ``descriptors_list`` is not a list or ``"all"``.
    """
    all_descriptor_funcs = dict(Descriptors.descList)

    if descriptors_list == "all":
        selected_descriptors = all_descriptor_funcs
    elif isinstance(descriptors_list, list):
        selected_descriptors = {
            name: all_descriptor_funcs[name]
            for name in descriptors_list
            if name in all_descriptor_funcs
        }
        missing = set(descriptors_list) - set(selected_descriptors)
        if missing:
            logger.warning(
                f"The following RDKit descriptors were not found and will be "
                f"skipped: {sorted(missing)}."
            )
    else:
        raise ValueError("descriptors_list must be a list of descriptor name strings or 'all'.")

    descriptors: dict[str, list[float]] = {}
    for smiles in smiles_list:
        mol = MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Could not parse SMILES '{smiles}' — skipping.")
            continue
        descriptors[smiles] = [func(mol) for func in selected_descriptors.values()]

    return descriptors, list(selected_descriptors.keys())


def get_polybert_fingerprints(psmiles_list: list[str]) -> dict[str, list[float]]:
    """
    Compute PolyBERT fingerprints for a list of PSMILES strings.

    Requires the ``psmiles`` package and a compatible PolyBERT model.

    Parameters
    ----------
    psmiles_list:
        PSMILES strings to compute fingerprints for.

    Returns
    -------
    dict[str, list[float]]
        Mapping from PSMILES string to its PolyBERT fingerprint vector.
    """
    try:
        from psmiles import PolymerSMILES as PS
    except ImportError as e:
        raise ImportError(
            "The 'psmiles' package is required for PolyBERT fingerprints. "
            "Install it with: pip install psmiles"
        ) from e

    fingerprints: dict[str, list[float]] = {}
    for psmile in psmiles_list:
        fingerprints[psmile] = PS(psmile).fingerprint("polyBERT")

    return fingerprints


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_unique_smiles(data: pd.DataFrame, smiles_cols: list[str]) -> list[str]:
    """Return all unique SMILES strings across all SMILES columns."""
    unique: list[str] = []
    for col in smiles_cols:
        unique.extend(data[col].dropna().unique().tolist())
    return list(dict.fromkeys(unique))  # preserve order, deduplicate


def _calculate_rdkit_df_dict(
    unique_smiles: list[str],
    data: pd.DataFrame,
    smiles_cols: list[str],
    rdkit_descriptors: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Compute RDKit descriptors once for unique SMILES and join to each
    SMILES column in the dataset.
    """
    desc_dict, columns = calculate_descriptors(unique_smiles, rdkit_descriptors)
    descriptors_df = pd.DataFrame.from_dict(desc_dict, orient="index", columns=columns)

    rdkit_df_dict: dict[str, pd.DataFrame] = {}
    for col in smiles_cols:
        joined = data.join(descriptors_df, how="left", on=col)
        rdkit_df_dict[col] = joined.drop(columns=data.columns)

    return rdkit_df_dict


def _merge(
    df_dict: dict[str, pd.DataFrame],
    data: pd.DataFrame,
    weights_col: dict[str, str] | None,
    data_index: pd.DataFrame,
    merging_approach: DescriptorMergingMethod,
) -> pd.DataFrame:
    """Dispatch to the correct merging function."""
    match merging_approach:
        case DescriptorMergingMethod.WeightedAverage:
            return _merge_weighted(df_dict, data, weights_col, data_index)
        case DescriptorMergingMethod.Average:
            return _merge_average(df_dict, data_index)
        case DescriptorMergingMethod.Concatenate:
            return _merge_concatenate(df_dict, data_index)
        case DescriptorMergingMethod.NoMerging:
            return _single_smiles(df_dict, data_index)
        case _:
            raise ValueError(
                f"Unsupported merging method: '{merging_approach}'. "
                f"Available: {[m.value for m in DescriptorMergingMethod]}."
            )


def _merge_weighted(
    df_dict: dict[str, pd.DataFrame],
    data: pd.DataFrame,
    weights_col: dict[str, str],
    data_index: pd.DataFrame,
) -> pd.DataFrame:
    weighted = {
        col: df.multiply(data[weights_col[col]], axis=0) / 100 for col, df in df_dict.items()
    }
    combined = sum(weighted.values())
    return pd.concat([data_index, combined], axis=1)


def _merge_average(df_dict: dict[str, pd.DataFrame], data_index: pd.DataFrame) -> pd.DataFrame:
    avg = sum(df_dict.values()) / len(df_dict)
    return pd.concat([data_index, avg], axis=1)


def _merge_concatenate(df_dict: dict[str, pd.DataFrame], data_index: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        key: df.rename(columns={col: f"{key}_{col}" for col in df.columns})
        for key, df in df_dict.items()
    }
    combined = pd.concat(renamed.values(), axis=1)
    return pd.concat([data_index, combined], axis=1)


def _single_smiles(df_dict: dict[str, pd.DataFrame], data_index: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([data_index] + list(df_dict.values()), axis=1)
