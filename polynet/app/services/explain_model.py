from collections import defaultdict
import json
from pathlib import Path

import captum
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
from torch_geometric.explain import CaptumExplainer, Explainer, GNNExplainer, ModelConfig
from torch_geometric.loader import DataLoader

from polynet.app.options.file_paths import (
    explanation_json_file_path,
    explanation_parent_directory,
    explanation_plots_path,
)
from polynet.app.utils import filter_dataset_by_ids
from polynet.config.constants import ResultColumn
from polynet.utils.chem_utils import fragment_and_match
from polynet.config.enums import (
    AtomBondDescriptorDictKey,
    DimensionalityReduction,
    ExplainAlgorithm,
    ImportanceNormalisationMethod,
    ProblemType,
)
from polynet.explainability import (
    calculate_masking_attributions,
    fragment_attributions_to_distribution,
    get_fragment_importance,
    merge_fragment_attributions,
    plot_attribution_distribution,
    plot_mols_with_numeric_weights,
    plot_mols_with_weights,
)
from polynet.featurizer.polymer_graph import CustomPolymerGraph

# Define a softer blue and red


def get_cmap(neg_color="#40bcde", pos_color="#e64747"):
    """
    Create a custom colormap with softer blue and red colors.
    Args:
        neg_color (str): Hex color code for the negative class (default is a soft blue).
        pos_color (str): Hex color code for the positive class (default is a soft red).
    Returns:
        LinearSegmentedColormap: A colormap that transitions from soft blue to white to soft red.
    """

    # Create a new colormap with less intense colors
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "soft_bwr", [neg_color, "white", pos_color]
    )

    return custom_cmap


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


def explain_model(
    models: dict[str, Module],
    model_number: int,
    experiment_path: Path,
    dataset: CustomPolymerGraph,
    explain_mols: list,
    plot_mols: list,
    explain_algorithm: ExplainAlgorithm,
    problem_type: ProblemType,
    neg_color: str = "#40bcde",
    pos_color: str = "#e64747",
    normalisation_type: str = "local",
    cutoff_explain: float = 0.1,
    mol_names: dict = {},
    predictions: dict = {},
    node_features: dict = {},
    explain_feature: str = "All Features",
    fragmentation_approach: str = "brics",
    target_class: int | None = None,
):

    # get colormap for visualization
    cmap = get_cmap(neg_color=neg_color, pos_color=pos_color)

    # Chemistry-aware masking is Captum-free — branch off before any Captum setup
    if explain_algorithm == ExplainAlgorithm.ChemistryMasking:
        _explain_model_masking(
            models=models,
            experiment_path=experiment_path,
            dataset=dataset,
            explain_mols=explain_mols,
            plot_mols=plot_mols,
            neg_color=neg_color,
            pos_color=pos_color,
            problem_type=problem_type,
            mol_names=mol_names,
            predictions=predictions,
            fragmentation_approach=fragmentation_approach,
            target_class=target_class,
        )
        return


def calculate_attributions(
    mols: list,
    existing_explanations: dict,
    explain_algorithm: ExplainAlgorithm,
    explainers: dict[str, Explainer],
):
    node_masks = {}

    for model_name_number, explainer in explainers.items():

        model_name, model_number = model_name_number.split("_")
        attrs_mol = existing_explanations.get(model_name, {}).get(str(model_number), {})

        for mol in mols:
            mol_idx = mol.idx

            if (
                attrs_mol is not None
                and mol_idx in attrs_mol
                and explain_algorithm in attrs_mol[mol_idx]
            ):
                node_mask = attrs_mol[mol_idx][explain_algorithm]
            else:
                node_mask = (
                    explainer(
                        x=mol.x,
                        edge_index=mol.edge_index,
                        batch_index=None,
                        edge_attr=mol.edge_attr,
                        monomer_weight=getattr(mol, "weight_monomer", None),
                        index=0,
                    )
                    .node_mask.detach()
                    .numpy()
                    .tolist()
                )

            # Insert in nested structure
            node_masks.setdefault(model_name, {}).setdefault(model_number, {}).setdefault(
                mol_idx, {}
            )[explain_algorithm] = node_mask

    return node_masks


def get_graph_embeddings(dataset: CustomPolymerGraph, model) -> np.ndarray:
    """
    Get graph embeddings for the dataset using the provided model.

    Args:
        dataset (CustomPolymerGraph): The dataset containing graph data.
        model: The model used to generate embeddings.

    Returns:
        np.ndarray: An array of graph embeddings.
    """
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
    embeddings = pd.DataFrame(np.concatenate(embeddings, axis=0), index=idx)

    return embeddings


def plot_projection_embeddings(
    tsne_embeddings: np.ndarray,
    labels: list = None,
    cmap: str = "blues",
    style: list = None,
    color_by_name: str = None,
    title: str = "Projection of Graph Embeddings",
) -> plt.Figure:
    """
    Plot projection of embeddings using seaborn (simplified version).

    Args:
        tsne_embeddings: 2D array of embeddings
        labels: Color mapping (optional)
        cmap: Colormap name (default: "blues")
        markers: List of marker styles for each point (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=400)

    sns.scatterplot(
        x=tsne_embeddings[:, 0],
        y=tsne_embeddings[:, 1],
        hue=labels,
        style=style,
        palette=cmap,
        s=50,  # Adjust point size as needed
    )

    plt.title(title, fontsize=25)
    plt.xlabel("Component 1", fontsize=22)
    plt.ylabel("Component 2", fontsize=22)
    plt.grid()

    is_continuous = (
        labels is not None
        and np.issubdtype(np.array(labels).dtype, np.number)
        and len(np.unique(labels)) > 10  # Arbitrary threshold for "continuous"
    )

    if is_continuous:
        norm = Normalize(vmin=min(labels), vmax=max(labels))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=color_by_name)  # Explicitly pass `ax`
        plt.gca().get_legend().remove()

    return fig


def deep_update(original: dict, new: dict):
    """Recursively update nested dictionaries."""
    for key, value in new.items():
        if isinstance(value, dict) and key in original and isinstance(original[key], dict):
            deep_update(original[key], value)
        else:
            original[key] = value
    return original


def _fragment_attributions_to_atom_weights(
    mol, monomer_dict: dict, frag_key: str, fragmentation_approach
) -> list[list[float]]:
    """
    Map per-fragment masking attributions back to per-atom weights.

    For each monomer in ``mol.mols``, each atom is assigned the attribution
    score of the fragment occurrence it belongs to (fragments are
    non-overlapping, so every atom maps to exactly one occurrence).
    Atoms with no matching fragment receive 0.0.

    Parameters
    ----------
    mol:
        PyG graph object; its ``.mols`` attribute lists monomer SMILES strings.
    monomer_dict:
        ``{monomer_smiles: {frag_key: {frag_smiles: [score_occ0, ...]}}}``,
        as produced by ``merge_fragment_attributions`` for a single molecule.
    frag_key:
        String key for the fragmentation method (e.g. ``"brics"``).
    fragmentation_approach:
        Fragmentation strategy passed to ``fragment_and_match``.

    Returns
    -------
    list[list[float]]
        One weight list per entry in ``mol.mols``, each of length
        ``rdkit_mol.GetNumAtoms()``.
    """
    weights_list = []
    for smiles in mol.mols:
        rdkit_mol = Chem.MolFromSmiles(smiles)
        if rdkit_mol is None:
            weights_list.append([])
            continue

        n_atoms = rdkit_mol.GetNumAtoms()
        weights = [0.0] * n_atoms

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


def _explain_model_masking(
    models: dict,
    experiment_path: Path,
    dataset,
    explain_mols: list,
    plot_mols: list,
    neg_color: str,
    pos_color: str,
    problem_type,
    mol_names: dict,
    predictions: dict,
    fragmentation_approach: str,
    target_class: int | None = None,
):
    """
    Captum-free explainability using chemistry-aware node masking.

    Computes fragment attributions as Y_pred_full − Y_pred_masked, where
    fragment nodes are zeroed in the pre-pooling embedding space.
    """
    import pandas as pd

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

    # Compute masking attributions (Captum-free)
    node_masks = calculate_masking_attributions(
        mols=mols,
        models=models,
        fragmentation_method=fragmentation_approach,
        problem_type=problem_type,
        existing_explanations=existing_explanations,
        target_class=target_class,
    )

    # Persist to cache
    combined_explanations = deep_update(existing_explanations, node_masks)
    with open(explanation_file, "w") as f:
        json.dump(combined_explanations, f, indent=4)

    # Merge across ensemble models (average per fragment, skip missing)
    merged = merge_fragment_attributions(node_masks, list(models.keys()))

    # Stable string keys (mirror masking._class_key and FragmentationMethod.value)
    class_key = "regression" if target_class is None else str(target_class)
    from polynet.config.enums import FragmentationMethod as _FM

    frag_key = (
        fragmentation_approach.value
        if isinstance(fragmentation_approach, _FM)
        else str(fragmentation_approach)
    )

    # Collect per-fragment score distributions for the ridge plot
    frags_importances = fragment_attributions_to_distribution(
        merged, target_class, fragmentation_approach
    )

    fig = plot_attribution_distribution(
        attribution_dict=frags_importances, neg_color=neg_color, pos_color=pos_color
    )
    st.pyplot(fig, use_container_width=True)

    # Per-molecule fragment score tables + atom colourmap
    plot_mols_filtered = filter_dataset_by_ids(dataset, plot_mols)

    for mol in plot_mols_filtered:
        container = st.container(border=True, key=f"mol_{mol.idx}_masking_container")
        container.info(
            f"Fragment attributions for `{mol.idx}` — Chemistry Masking "
            f"({frag_key})" + (f", class `{target_class}`" if target_class is not None else "")
        )
        container.write(
            f"True label: `{predictions.get(mol.idx, {}).get(ResultColumn.LABEL, 'N/A')}`"
        )
        container.write(
            f"Predicted label: `{predictions.get(mol.idx, {}).get(ResultColumn.PREDICTED, 'N/A')}`"
        )

        # {monomer_smiles: {frag_key: {frag_smiles: [scores]}}}
        monomer_dict = (
            merged.get(mol.idx, {})
            .get(ExplainAlgorithm.ChemistryMasking.value, {})
            .get(class_key, {})
        )

        if monomer_dict:
            rows = [
                {
                    "Monomer (SMILES)": monomer_smiles,
                    "Fragment (SMILES)": frag_smiles,
                    "Occurrence": occ + 1,
                    "Attribution (Y − Y_masked)": score,
                }
                for monomer_smiles, fk_dict in monomer_dict.items()
                for frag_smiles, scores in fk_dict.get(frag_key, {}).items()
                for occ, score in enumerate(scores)
            ]
            df = pd.DataFrame(rows).sort_values("Attribution (Y − Y_masked)", ascending=False)
            container.dataframe(df, use_container_width=True)

            # Map fragment attributions back to atom-level weights
            weights_list = _fragment_attributions_to_atom_weights(
                mol=mol,
                monomer_dict=monomer_dict,
                frag_key=frag_key,
                fragmentation_approach=fragmentation_approach,
            )

            # Normalise locally (divide by max absolute weight across all monomers)
            all_weights = [w for wlist in weights_list for w in wlist]
            max_abs = max((abs(w) for w in all_weights), default=1.0) or 1.0
            weights_list = [[w / max_abs for w in wlist] for wlist in weights_list]

            names = mol_names.get(mol.idx, None)
            cmap = get_cmap(neg_color=neg_color, pos_color=pos_color)
            fig = plot_mols_with_weights(
                smiles_list=mol.mols,
                weights_list=weights_list,
                colormap=cmap,
                legend=names,
                min_weight=-1.0,
                max_weight=1.0,
            )
            container.pyplot(fig, use_container_width=True)
        else:
            container.warning("No fragments matched for this molecule.")
