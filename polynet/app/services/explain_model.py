import json
from pathlib import Path

import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import streamlit as st
import torch
from torch.nn import Module
from torch_geometric.loader import DataLoader

from polynet.app.options.file_paths import explanation_json_file_path, explanation_parent_directory
from polynet.app.utils import filter_dataset_by_ids
from polynet.config.constants import ResultColumn
from polynet.utils.chem_utils import fragment_and_match
from polynet.config.enums import (
    AttributionPlotType,
    DimensionalityReduction,
    ExplainAlgorithm,
    FragmentationMethod,
    ImportanceNormalisationMethod,
    ProblemType,
)
from polynet.explainability import (
    calculate_masking_attributions,
    fragment_attributions_to_distribution,
    merge_fragment_attributions,
    plot_attribution_bar,
    plot_attribution_distribution,
    plot_attribution_strip,
    plot_mols_with_weights,
)
from polynet.featurizer.polymer_graph import CustomPolymerGraph


# ---------------------------------------------------------------------------
# Colourmap helper
# ---------------------------------------------------------------------------


def get_cmap(neg_color="#40bcde", pos_color="#e64747"):
    """Create a custom diverging colormap (blue → white → red)."""
    return mcolors.LinearSegmentedColormap.from_list("soft_bwr", [neg_color, "white", pos_color])


# ---------------------------------------------------------------------------
# Graph embedding visualisation
# ---------------------------------------------------------------------------


def analyse_graph_embeddings(
    model,
    dataset: CustomPolymerGraph,
    labels: pd.Series,
    label_name: str,
    style_by: pd.Series,
    mols_to_plot: list,
    reduction_method: str,
    reduction_parameters: dict,
    colormap: str,
):
    embeddings = get_graph_embeddings(dataset, model)

    if reduction_method == DimensionalityReduction.tSNE:
        tsne = TSNE(n_components=2, **reduction_parameters)
        reduced = tsne.fit_transform(embeddings)
    elif reduction_method == DimensionalityReduction.PCA:
        pca = PCA(n_components=2, **reduction_parameters)
        reduced = pca.fit_transform(embeddings)

    reduced_embeddings = pd.DataFrame(reduced, index=embeddings.index, columns=["Dim1", "Dim2"])
    reduced_embeddings = reduced_embeddings.loc[mols_to_plot]

    embedding_table = pd.concat([reduced_embeddings, labels, style_by], axis=1)
    reduced_embeddings = reduced_embeddings.to_numpy()
    labels = labels.loc[mols_to_plot]

    projection_fig = plot_projection_embeddings(
        reduced_embeddings, labels=labels, cmap=colormap, style=style_by, color_by_name=label_name
    )
    st.pyplot(projection_fig, use_container_width=True)

    if st.checkbox("Show embedding data table"):
        st.write(embedding_table)


def get_graph_embeddings(dataset: CustomPolymerGraph, model) -> pd.DataFrame:
    """Extract graph-level embeddings for every molecule in the dataset."""
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    embeddings = []
    idx = []

    for batch in loader:
        with torch.no_grad():
            embedding = model.get_graph_embedding(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch_index=batch.batch,
                monomer_weight=getattr(batch, "weight_monomer", None),
            )
            embeddings.append(embedding.cpu().numpy())
            idx.append(batch.idx)

    idx = np.array(idx).flatten().tolist()
    return pd.DataFrame(np.concatenate(embeddings, axis=0), index=idx)


def plot_projection_embeddings(
    tsne_embeddings: np.ndarray,
    labels: list = None,
    cmap: str = "blues",
    style: list = None,
    color_by_name: str = None,
    title: str = "Projection of Graph Embeddings",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8), dpi=400)

    sns.scatterplot(
        x=tsne_embeddings[:, 0],
        y=tsne_embeddings[:, 1],
        hue=labels,
        style=style,
        palette=cmap,
        s=50,
    )

    plt.title(title, fontsize=25)
    plt.xlabel("Component 1", fontsize=22)
    plt.ylabel("Component 2", fontsize=22)
    plt.grid()

    is_continuous = (
        labels is not None
        and np.issubdtype(np.array(labels).dtype, np.number)
        and len(np.unique(labels)) > 10
    )

    if is_continuous:
        norm = Normalize(vmin=min(labels), vmax=max(labels))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=color_by_name)
        plt.gca().get_legend().remove()

    return fig


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def deep_update(original: dict, new: dict):
    """Recursively update nested dictionaries in-place; returns original."""
    for key, value in new.items():
        if isinstance(value, dict) and key in original and isinstance(original[key], dict):
            deep_update(original[key], value)
        else:
            original[key] = value
    return original


def _frag_key(fragmentation_approach) -> str:
    return (
        fragmentation_approach.value
        if isinstance(fragmentation_approach, FragmentationMethod)
        else str(fragmentation_approach)
    )


def _class_key(target_class: int | None) -> str:
    return "regression" if target_class is None else str(target_class)


def _build_display_data(combined_explanations: dict, models: dict, mol_ids: list) -> dict:
    """
    Filter the full cache to only the requested models and molecule IDs.

    JSON keys are always strings; mol_ids are str-normalised for the lookup
    so integer IDs from pandas indices match correctly.
    """
    mol_id_set = set(str(m) for m in mol_ids)
    display_data: dict = {}
    for model_log_name in models.keys():
        model_name, model_number = model_log_name.split("_", 1)
        mol_cache = combined_explanations.get(model_name, {}).get(model_number, {})
        for mol_id, mol_entry in mol_cache.items():
            if str(mol_id) in mol_id_set:
                (display_data.setdefault(model_name, {}).setdefault(model_number, {}))[
                    mol_id
                ] = mol_entry
    return display_data


def _fragment_attributions_to_atom_weights(
    mol, monomer_dict: dict, frag_key: str, fragmentation_approach
) -> list[list[float]]:
    """Map per-fragment masking attributions back to per-atom weights."""
    weights_list = []
    for smiles in mol.mols:
        rdkit_mol = Chem.MolFromSmiles(smiles)
        if rdkit_mol is None:
            weights_list.append([])
            continue

        weights = [0.0] * rdkit_mol.GetNumAtoms()
        frag_scores = monomer_dict.get(smiles, {}).get(frag_key, {})
        if frag_scores:
            frags = fragment_and_match(smiles, fragmentation_approach)
            for frag_smiles, atom_indices_list in frags.items():
                scores = frag_scores.get(frag_smiles, [])
                for occ_idx, atom_indices in enumerate(atom_indices_list):
                    if occ_idx < len(scores):
                        score = scores[occ_idx]
                        for atom_idx in atom_indices:
                            weights[atom_idx] = score
        weights_list.append(weights)
    return weights_list


# ---------------------------------------------------------------------------
# Shared computation: compute + cache masking attributions
# ---------------------------------------------------------------------------


def _compute_and_cache_masking(
    models: dict,
    experiment_path: Path,
    dataset,
    explain_mols: list,
    problem_type: ProblemType,
    fragmentation_approach,
    target_class: int | None = None,
) -> dict:
    """
    Load the JSON cache, compute any missing masking attributions, persist,
    and return the full combined_explanations dict.

    Raw scores are always stored; normalisation is never written to the cache.
    Callers should call ``_build_display_data`` on the returned dict to obtain
    a view filtered to their requested models and molecules.
    """
    explain_path = explanation_parent_directory(experiment_path)
    if not explain_path.exists():
        explain_path.mkdir(parents=True, exist_ok=True)

    explanation_file = explanation_json_file_path(experiment_path=experiment_path)
    if explanation_file.exists():
        with open(explanation_file) as f:
            existing_explanations = json.load(f)
    else:
        existing_explanations = {}

    if isinstance(explain_mols, str):
        explain_mols = [explain_mols]

    mols = filter_dataset_by_ids(dataset, explain_mols)

    node_masks = calculate_masking_attributions(
        mols=mols,
        models=models,
        fragmentation_method=fragmentation_approach,
        problem_type=problem_type,
        existing_explanations=existing_explanations,
        target_class=target_class,
    )

    combined_explanations = deep_update(existing_explanations, node_masks)
    with open(explanation_file, "w") as f:
        json.dump(combined_explanations, f, indent=4)

    return combined_explanations


# ---------------------------------------------------------------------------
# Global explanation: population-level fragment distribution
# ---------------------------------------------------------------------------


def explain_model_global(
    models: dict,
    experiment_path: Path,
    dataset,
    explain_mols: list,
    problem_type: ProblemType,
    neg_color: str = "#40bcde",
    pos_color: str = "#e64747",
    normalisation_type: ImportanceNormalisationMethod = ImportanceNormalisationMethod.PerModel,
    fragmentation_approach=FragmentationMethod.BRICS,
    target_class: int | None = None,
    top_n: int | None = None,
    plot_type: AttributionPlotType = AttributionPlotType.Ridge,
) -> None:
    """
    Render the population-level fragment attribution plot.

    Shows which fragments drive model predictions across the selected set of
    molecules.  Every individual model score is preserved so the distribution
    reflects both model uncertainty and molecule variability.

    ``plot_type`` controls the visualisation style:
    - Ridge  — overlapping KDE rows (full distribution per fragment)
    - Bar    — mean ± 95 % CI horizontal bars
    - Strip  — individual scores jittered per row, mean overlaid as ◆
    """
    combined_explanations = _compute_and_cache_masking(
        models=models,
        experiment_path=experiment_path,
        dataset=dataset,
        explain_mols=explain_mols,
        problem_type=problem_type,
        fragmentation_approach=fragmentation_approach,
        target_class=target_class,
    )

    display_data = _build_display_data(combined_explanations, models, explain_mols)

    fk = _frag_key(fragmentation_approach)
    ck = _class_key(target_class)
    _ALG_KEY = ExplainAlgorithm.ChemistryMasking.value

    # Count how many fragments are available before top_n filtering
    all_frags = {
        frag
        for model_data in display_data.values()
        for num_data in model_data.values()
        for mol_data in num_data.values()
        for mon_data in mol_data.get(_ALG_KEY, {}).get(ck, {}).values()
        for frag in mon_data.get(fk, {})
    }
    n_mols = sum(
        len(num_data) for model_data in display_data.values() for num_data in model_data.values()
    )
    n_models = len(models)
    n_frags_total = len(all_frags)
    shown = min(top_n * 2, n_frags_total) if top_n and n_frags_total > top_n * 2 else n_frags_total

    st.info(
        f"Distribution over **{n_mols}** molecule(s) × **{n_models}** model(s) — "
        f"**{n_frags_total}** unique fragments found, showing top/bottom **{shown // 2}** each."
        + (f" | class `{target_class}`" if target_class is not None else "")
        + f" | normalisation: `{normalisation_type}`"
    )

    frags_importances_display = fragment_attributions_to_distribution(
        node_masks=display_data,
        model_log_names=list(models.keys()),
        target_class=target_class,
        fragmentation_method=fragmentation_approach,
        normalisation_type=normalisation_type,
    )

    if not frags_importances_display:
        st.warning("No fragment attributions found for the selected molecules and models.")
        return

    shared_kwargs = dict(
        attribution_dict=frags_importances_display,
        neg_color=neg_color,
        pos_color=pos_color,
        top_n=top_n,
    )

    if plot_type == AttributionPlotType.Bar:
        fig = plot_attribution_bar(**shared_kwargs)
    elif plot_type == AttributionPlotType.Strip:
        fig = plot_attribution_strip(**shared_kwargs)
    else:  # Ridge (default)
        fig = plot_attribution_distribution(**shared_kwargs)

    st.pyplot(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Local explanation: per-molecule attribution panels
# ---------------------------------------------------------------------------


def explain_model_local(
    models: dict,
    experiment_path: Path,
    dataset,
    explain_mols: list,
    problem_type: ProblemType,
    neg_color: str = "#40bcde",
    pos_color: str = "#e64747",
    normalisation_type: ImportanceNormalisationMethod = ImportanceNormalisationMethod.PerModel,
    fragmentation_approach=FragmentationMethod.BRICS,
    target_class: int | None = None,
    mol_names: dict | None = None,
    predictions: dict | None = None,
    class_labels: dict | None = None,
) -> None:
    """
    Render per-molecule attribution panels (table + atom heatmap).

    ``explain_mols`` is the display set — every molecule in this list gets its
    own bordered container with a fragment attribution table and atom colourmap.
    """
    mol_names = mol_names or {}
    predictions = predictions or {}
    class_labels = class_labels or {}

    combined_explanations = _compute_and_cache_masking(
        models=models,
        experiment_path=experiment_path,
        dataset=dataset,
        explain_mols=explain_mols,
        problem_type=problem_type,
        fragmentation_approach=fragmentation_approach,
        target_class=target_class,
    )

    display_data = _build_display_data(combined_explanations, models, explain_mols)

    fk = _frag_key(fragmentation_approach)
    ck = _class_key(target_class)
    _ALG_KEY = ExplainAlgorithm.ChemistryMasking.value

    merged = merge_fragment_attributions(display_data, list(models.keys()))

    # Global divisor derived from the averaged merged data (not individual model scores)
    if normalisation_type == ImportanceNormalisationMethod.Global:
        global_divisor_mol = (
            max(
                (
                    abs(s)
                    for mol_data in merged.values()
                    for mon_data in mol_data.get(_ALG_KEY, {}).get(ck, {}).values()
                    for scores in mon_data.get(fk, {}).values()
                    for s in scores
                ),
                default=1.0,
            )
            or 1.0
        )
    else:
        global_divisor_mol = None

    mols_filtered = filter_dataset_by_ids(dataset, explain_mols)
    cmap = get_cmap(neg_color=neg_color, pos_color=pos_color)

    for mol in mols_filtered:
        container = st.container(border=True, key=f"local_mol_{mol.idx}_container")
        container.info(
            f"Fragment attributions for `{mol.idx}` — Chemistry Masking ({fk})"
            + (f", class `{target_class}`" if target_class is not None else "")
            + f" | normalisation: `{normalisation_type}`"
        )
        container.write(
            f"True label: `{predictions.get(mol.idx, {}).get(ResultColumn.LABEL, 'N/A')}`"
        )
        container.write(
            f"Predicted label: `{predictions.get(mol.idx, {}).get(ResultColumn.PREDICTED, 'N/A')}`"
        )

        monomer_dict = merged.get(mol.idx, {}).get(_ALG_KEY, {}).get(ck, {})
        if not monomer_dict:
            container.warning("No fragments matched for this molecule.")
            continue

        mol_raw_scores = [
            s
            for mon_data in monomer_dict.values()
            for scores in mon_data.get(fk, {}).values()
            for s in scores
        ]

        if normalisation_type in (
            ImportanceNormalisationMethod.Local,
            ImportanceNormalisationMethod.PerModel,
        ):
            mol_divisor = max((abs(s) for s in mol_raw_scores), default=1.0) or 1.0
        elif normalisation_type == ImportanceNormalisationMethod.Global:
            mol_divisor = global_divisor_mol
        else:
            mol_divisor = None

        attr_col = "Attribution (Y − Y_masked)"
        norm_col = "Normalised Attribution" if mol_divisor is not None else attr_col
        rows = [
            {
                "Monomer (SMILES)": monomer_smiles,
                "Fragment (SMILES)": frag_smiles,
                "Occurrence": occ + 1,
                attr_col: score,
            }
            for monomer_smiles, fk_dict in monomer_dict.items()
            for frag_smiles, scores in fk_dict.get(fk, {}).items()
            for occ, score in enumerate(scores)
        ]
        df = pd.DataFrame(rows).sort_values(attr_col, ascending=False)
        if mol_divisor is not None:
            df[norm_col] = df[attr_col] / mol_divisor
        container.dataframe(df, use_container_width=True)

        weights_list = _fragment_attributions_to_atom_weights(
            mol=mol,
            monomer_dict=monomer_dict,
            frag_key=fk,
            fragmentation_approach=fragmentation_approach,
        )

        if mol_divisor is not None:
            weights_list = [[w / mol_divisor for w in wlist] for wlist in weights_list]
            cmap_min, cmap_max = -1.0, 1.0
        else:
            cmap_min, cmap_max = None, None

        names = mol_names.get(mol.idx, None)
        fig = plot_mols_with_weights(
            smiles_list=mol.mols,
            weights_list=weights_list,
            colormap=cmap,
            legend=names,
            min_weight=cmap_min,
            max_weight=cmap_max,
        )
        container.pyplot(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Legacy entry point (backward compatibility)
# ---------------------------------------------------------------------------


def explain_model(
    models: dict[str, Module],
    experiment_path: Path,
    dataset: CustomPolymerGraph,
    explain_mols: list,
    plot_mols: list,
    explain_algorithm: ExplainAlgorithm,
    problem_type: ProblemType,
    neg_color: str = "#40bcde",
    pos_color: str = "#e64747",
    normalisation_type=ImportanceNormalisationMethod.PerModel,
    cutoff_explain: float = 0.1,
    mol_names: dict = {},
    predictions: dict = {},
    fragmentation_approach=FragmentationMethod.BRICS,
    target_class: int | None = None,
    top_n: int | None = None,
) -> None:
    """Legacy entry point — runs global then local in sequence."""
    if explain_algorithm == ExplainAlgorithm.ChemistryMasking:
        explain_model_global(
            models=models,
            experiment_path=experiment_path,
            dataset=dataset,
            explain_mols=explain_mols,
            problem_type=problem_type,
            neg_color=neg_color,
            pos_color=pos_color,
            normalisation_type=normalisation_type,
            fragmentation_approach=fragmentation_approach,
            target_class=target_class,
            top_n=top_n,
        )
        if plot_mols:
            explain_model_local(
                models=models,
                experiment_path=experiment_path,
                dataset=dataset,
                explain_mols=plot_mols,
                problem_type=problem_type,
                neg_color=neg_color,
                pos_color=pos_color,
                normalisation_type=normalisation_type,
                fragmentation_approach=fragmentation_approach,
                target_class=target_class,
                mol_names=mol_names,
                predictions=predictions,
            )
