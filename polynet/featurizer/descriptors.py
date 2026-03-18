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
from polymetrix.featurizers.polymer import Polymer
from rdkit.Chem import Descriptors, MolFromSmiles

from polynet.config.column_names import get_fp_col_names
from polynet.config.enums import DescriptorMergingMethod, MolecularDescriptor
from polynet.data.preprocessing import get_data_index
from polynet.featurizer.pmx import create_pmx_featurizer

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
    unique_smiles = get_unique_smiles(data, smiles_cols)

    descriptors = {}

    # --- RDKit descriptors ---
    rdkit_descriptors_df: pd.DataFrame | None = None
    rdkit_desc_list = molecular_descriptors.get(MolecularDescriptor.RDKit, [])

    if rdkit_desc_list and (rdkit_independent or mix_rdkit_df_descriptors):
        rdkit_df_dict = calculate_rdkit_df_dict(unique_smiles, data, smiles_cols, rdkit_desc_list)
        rdkit_descriptors_df = _merge(
            rdkit_df_dict, data, weights_col, data_index, merging_approach
        )
        descriptors[MolecularDescriptor.RDKit] = rdkit_descriptors_df

    if not rdkit_independent:
        descriptors[MolecularDescriptor.RDKit] = None

    # --- DataFrame descriptors ---
    df_desc_cols = molecular_descriptors.get(MolecularDescriptor.DataFrame, [])
    df_descriptors_df: pd.DataFrame | None = None

    if df_desc_cols and df_descriptors_independent:
        descriptors_df = data[df_desc_cols].copy()
        df_descriptors_df = pd.concat([data_index, descriptors_df], axis=1)
        descriptors[MolecularDescriptor.DataFrame] = df_descriptors_df

    # --- Mixed RDKit + DataFrame ---
    mixed_df: pd.DataFrame | None = None
    if mix_rdkit_df_descriptors and rdkit_descriptors_df is not None and df_desc_cols:
        descriptors_df = data[df_desc_cols].copy()
        mixed_df = pd.concat([rdkit_descriptors_df, descriptors_df], axis=1)
        descriptors[MolecularDescriptor.RDKit_DataFrame] = mixed_df
    # --- PolyBERT ---
    if MolecularDescriptor.PolyBERT in molecular_descriptors:
        try:
            polyBERT_dict = calculate_polybert_df_dict(
                unique_psmiles=unique_smiles, data=data, psmiles_cols=smiles_cols
            )
            polyBERT_df = _merge(
                df_dict=polyBERT_dict,
                data=data,
                weights_col=weights_col,
                data_index=data_index,
                merging_approach=merging_approach,
            )
            descriptors[MolecularDescriptor.PolyBERT] = polyBERT_df
        except Exception as e:
            print(e)

    if MolecularDescriptor.PolyMetriX in molecular_descriptors:
        pmx_dict = calculate_PMX_df_dict(
            unique_psmiles=unique_smiles,
            data=data,
            psmiles_cols=smiles_cols,
            pmx_descriptors=molecular_descriptors[MolecularDescriptor.PolyMetriX],
        )
        pmx_df = _merge(
            df_dict=pmx_dict,
            data=data,
            weights_col=weights_col,
            data_index=data_index,
            merging_approach=merging_approach,
        )
        descriptors[MolecularDescriptor.PolyMetriX] = pmx_df

    return descriptors


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


def get_polybert_fingerprints(
    psmiles_list: list[str], batch_size: int = 64, show_progress_bar: bool = False
) -> dict[str, list[float]]:
    """
    Compute PolyBERT fingerprints for a list of PSMILES strings (batched + cached).

    Returns mapping: original_psmiles -> fingerprint vector (list[float])

    Notes
    -----
    - Canonicalization is applied before encoding.
    - Embeddings are computed only once per unique canonical PSMILES.
    - Output keys remain the original strings from `psmiles_list` so you can
      join back to your dataset without needing to canonicalize your dataframe.
    """
    from canonicalize_psmiles.canonicalize import canonicalize
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("kuelumbus/polyBERT")

    # 1) Canonicalize (keep a mapping from original -> canonical)
    orig_to_canon: dict[str, str] = {}
    canon_list: list[str] = []
    for p in psmiles_list:
        if p is None:
            continue
        p_str = str(p)
        try:
            c = canonicalize(p_str)
        except Exception:
            # If canonicalization fails, keep original; embedding may still work
            c = p_str
        orig_to_canon[p_str] = c
        canon_list.append(c)

    if not canon_list:
        return {}

    # 2) Deduplicate canonical strings (preserve order)
    seen = set()
    unique_canons: list[str] = []
    for c in canon_list:
        if c not in seen:
            seen.add(c)
            unique_canons.append(c)

    # 3) Batched encoding for unique canonical strings
    # convert_to_numpy=True returns np.ndarray directly
    embeddings = model.encode(
        unique_canons,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    # 4) Build canonical -> vector lookup (as python lists for JSON/pandas friendliness)
    canon_to_vec: dict[str, list[float]] = {
        c: embeddings[i].astype(float).tolist() for i, c in enumerate(unique_canons)
    }

    # 5) Map back to original strings
    return {orig: canon_to_vec[canon] for orig, canon in orig_to_canon.items()}


def calculate_polybert_df_dict(
    unique_psmiles: list[str], data: pd.DataFrame, psmiles_cols: list[str], prefix: str = "polybert"
) -> dict[str, pd.DataFrame]:
    """
    Compute PolyBERT fingerprints once for unique PSMILES and join to each
    PSMILES column in the dataset.

    Returns dict[col] -> DataFrame containing only the fingerprint columns
    aligned to `data`'s rows (via left join on that column).
    """
    fp_dict = get_polybert_fingerprints(unique_psmiles)

    if not fp_dict:
        # No valid fingerprints computed
        n_rows = len(data)
        out: dict[str, pd.DataFrame] = {}
        for col in psmiles_cols:
            out[col] = pd.DataFrame(index=data.index)
        return out

    # Determine embedding dimension from first vector
    first_vec = next(iter(fp_dict.values()))
    dim = len(first_vec)

    fp_cols = get_fp_col_names()
    fingerprints_df = pd.DataFrame.from_dict(fp_dict, orient="index", columns=fp_cols)

    polybert_df_dict: dict[str, pd.DataFrame] = {}
    for col in psmiles_cols:
        joined = data.join(fingerprints_df, how="left", on=col)
        polybert_df_dict[col] = joined[fp_cols].copy()

    return polybert_df_dict


def get_PMX_descriptors(
    unique_psmiles: list[str], side_chain_desc_list, backbone_desclist, agg_method
):

    feat_dict = {}

    featurizer = create_pmx_featurizer(
        side_chain_features=side_chain_desc_list,
        backbone_features=backbone_desclist,
        agg_method=agg_method,
    )

    for psmiles in unique_psmiles:
        polymer = Polymer.from_psmiles(psmiles)

        feat_dict[psmiles] = featurizer.featurize(polymer)

    return feat_dict, featurizer.feature_labels()


def calculate_PMX_df_dict(
    unique_psmiles: list[str], data: pd.DataFrame, psmiles_cols: list[str], pmx_descriptors: dict
):

    pmx_dict, cols = get_PMX_descriptors(
        unique_psmiles=unique_psmiles,
        side_chain_desc_list=pmx_descriptors["side_chain"],
        backbone_desclist=pmx_descriptors["backbone"],
        agg_method=pmx_descriptors["agg"],
    )

    pmx_df = pd.DataFrame.from_dict(pmx_dict, orient="index", columns=cols)

    pmx_df_dict: dict[str, pd.DataFrame] = {}
    for col in psmiles_cols:
        joined = data.join(pmx_df, how="left", on=col)
        pmx_df_dict[col] = joined[cols].copy()

    return pmx_df_dict


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


# TODO check if new function name with no underscore is adequate in this file
def get_unique_smiles(data: pd.DataFrame, smiles_cols: list[str]) -> list[str]:
    """Return all unique SMILES strings across all SMILES columns."""
    unique: list[str] = []
    for col in smiles_cols:
        unique.extend(data[col].dropna().unique().tolist())
    return list(dict.fromkeys(unique))  # preserve order, deduplicate


# TODO check if new function name with no underscore is adequate in this file
def calculate_rdkit_df_dict(
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
            return merge_weighted(df_dict, data, weights_col, data_index)
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


# TODO check if new function name with no underscore is adequate in this file
def merge_weighted(
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
