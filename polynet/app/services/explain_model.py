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
from polynet.config.enums import (
    AtomBondDescriptorDictKey,
    DimensionalityReduction,
    ExplainAlgorithm,
    ImportanceNormalisationMethod,
    ProblemType,
)
from polynet.explain.explain_mol import (
    get_fragment_importance,
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
):

    # get colormap for visualization
    cmap = get_cmap(neg_color=neg_color, pos_color=pos_color)

    # Set the problem type passed to the model config
    if problem_type == ProblemType.Classification:
        task = "multiclass_classification"
    elif problem_type == ProblemType.Regression:
        task = "regression"
    # Create the model configuration for explainer
    model_config = ModelConfig(mode=task, task_level="graph", return_type="raw")

    # Initialize the explainer based on the selected algorithm
    if explain_algorithm == ExplainAlgorithm.GNNExplainer:
        algorithm = GNNExplainer(model=model, epochs=100, return_type="raw", explain_graph=True)
    elif explain_algorithm == ExplainAlgorithm.ShapleyValueSampling:
        algorithm = CaptumExplainer(attribution_method=captum.attr.ShapleyValueSampling)
    elif explain_algorithm == ExplainAlgorithm.InputXGradients:
        algorithm = CaptumExplainer(attribution_method=captum.attr.InputXGradient)
    elif explain_algorithm == ExplainAlgorithm.Saliency:
        algorithm = CaptumExplainer(attribution_method=captum.attr.Saliency)
    elif explain_algorithm == ExplainAlgorithm.IntegratedGradients:
        algorithm = CaptumExplainer(attribution_method=captum.attr.IntegratedGradients)
    elif explain_algorithm == ExplainAlgorithm.Deconvolution:
        algorithm = CaptumExplainer(attribution_method=captum.attr.Deconvolution)
    elif explain_algorithm == ExplainAlgorithm.GuidedBackprop:
        algorithm = CaptumExplainer(attribution_method=captum.attr.GuidedBackprop)
    else:
        st.error(f"Unknown explain algorithm: {explain_algorithm}")
        return

    # Initialize the explainer with the model and algorithm
    explainers = {}

    for model_name, model in models.items():
        explainer = Explainer(
            model=model,
            algorithm=algorithm,
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type=None,
            model_config=model_config,
        )
        explainers[model_name] = explainer

    # Create the directory for explanations if it does not exist
    explain_path = explanation_parent_directory(experiment_path)
    if not explain_path.exists():
        explain_path.mkdir(parents=True, exist_ok=True)

    # Load existing explanations if available
    explanation_file = explanation_json_file_path(experiment_path=experiment_path)
    if explanation_file.exists():
        with open(explanation_file) as f:
            existing_explanations = json.load(f)
    else:
        existing_explanations = {}

    if isinstance(explain_mols, str):
        explain_mols = [explain_mols]

    # Filter the dataset to only include the molecules we want to explain
    mols = filter_dataset_by_ids(dataset, explain_mols)

    # Calculate the node masks for the selected molecules
    node_masks = calculate_attributions(
        mols=mols,
        existing_explanations=existing_explanations,
        explain_algorithm=explain_algorithm,
        explainers=explainers,
    )

    combined_explanations = deep_update(existing_explanations, node_masks)

    # --- Save merged explanations ---
    with open(explanation_file, "w") as f:
        json.dump(combined_explanations, f, indent=4)

    node_masks = merge_mask_dicts(node_masks, models.keys())

    # Check the selected feature for explanation
    if explain_feature != "All Features":
        feature_lengths = get_node_feat_vector_size(node_features)

        slicer_initial = 0
        slicer_final = 0

        for feat, length in feature_lengths.items():
            slicer_final += length
            if feat == explain_feature:
                break
            slicer_initial += slicer_final
    else:
        slicer_initial = 0
        slicer_final = None

    for mol in mols:
        mask = np.array(node_masks[mol.idx][explain_algorithm])
        if slicer_final is not None:
            mask = mask[:, slicer_initial:slicer_final]
        node_masks[mol.idx][explain_algorithm] = mask

    if normalisation_type == ImportanceNormalisationMethod.Global:

        max_val = None

        for mol in mols:
            maks = node_masks[mol.idx][explain_algorithm].sum(axis=1)
            local_max_val = np.max(np.abs(maks))
            if max_val is None or local_max_val > max_val:
                max_val = local_max_val
                st.write(
                    f"New global max value found: {max_val:.3f} for molecule {mol.idx} with algorithm {explain_algorithm}"
                )

    frags_importances = get_fragment_importance(
        mols=mols,
        node_masks=node_masks,
        explain_algorithm=explain_algorithm,
        fragmentation_approach=fragmentation_approach,
    )

    fig = plot_attribution_distribution(
        attribution_dict=frags_importances, neg_color=neg_color, pos_color=pos_color
    )
    st.pyplot(fig, use_container_width=True)

    plot_mols = filter_dataset_by_ids(dataset, plot_mols)

    for mol in plot_mols:

        container = st.container(border=True, key=f"mol_{mol.idx}_container")
        names = mol_names.get(mol.idx, None)
        masks = node_masks[mol.idx][explain_algorithm].sum(axis=1)

        if normalisation_type == ImportanceNormalisationMethod.Local:
            masks = masks / np.max(np.abs(masks))
        elif normalisation_type == ImportanceNormalisationMethod.Global:
            masks = masks / max_val
        else:
            pass

        masks = np.where(np.abs(masks) > cutoff_explain, masks, 0.0)
        masks = masks.tolist()

        masks_mol = []
        ia = 0
        for smiles in mol.mols:
            molecule = Chem.MolFromSmiles(smiles)
            n_atoms = molecule.GetNumAtoms()
            masks_mol.append(masks[ia : ia + n_atoms])
            ia += n_atoms

        container.info(f"Plotting molecule `{mol.idx}` with algorithm `{explain_algorithm}`")

        container.write(
            f"True label: `{predictions.get(mol.idx, {}).get(ResultColumn.Label, 'N/A')}`"
        )
        container.write(
            f"Predicted label: `{predictions.get(mol.idx, {}).get(ResultColumn.Predicted, 'N/A')}`"
        )

        fig = plot_mols_with_weights(
            smiles_list=mol.mols,
            weights_list=masks_mol,
            colormap=cmap,
            legend=names,
            min_weight=-1.0,
            max_weight=1.0,
        )
        container.pyplot(fig, use_container_width=True)

        fig = plot_mols_with_numeric_weights(
            smiles_list=mol.mols, weights_list=masks_mol, legend=names
        )
        container.pyplot(fig, use_container_width=True)


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
                        monomer_weight=mol.weight_monomer,
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


def get_node_feat_vector_size(node_features: dict) -> dict:

    lengths_dict = {}

    for key, value in node_features.items():
        if value == {}:
            lengths_dict[key] = 1
            continue
        allowed_features_size = len(value[AtomBondDescriptorDictKey.AllowableVals])
        wildcard_feat_size = int(value[AtomBondDescriptorDictKey.Wildcard])
        lengths_dict[key] = allowed_features_size + wildcard_feat_size
    return lengths_dict


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
                monomer_weight=batch.weight_monomer,
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


def merge_mask_dicts(node_masks, model_names_numbers):
    """
    Merge attribution masks from multiple models while normalizing each model/algorithm
    by the maximum absolute attribution observed across ALL molecules for that model+algorithm.

    Parameters
    ----------
    node_masks : dict
        Structure: node_masks[model_name][model_number][mol_id][algorithm] = list_of_feature_lists
    model_names_numbers : list[str]
        e.g. ["GNN_0", "GNN_1", "RF_0"]

    Returns
    -------
    dict
        merged result with same structure: {mol_id: {algorithm: averaged_mask_list}}
    """

    # Step 1: compute global max absolute per (model_name_number, algorithm)
    global_max = defaultdict(lambda: 0.0)
    for model_name_number in model_names_numbers:
        model_name, number = model_name_number.split("_")
        # defensive: skip if model or number missing
        if model_name not in node_masks or number not in node_masks[model_name]:
            continue
        model_dict = node_masks[model_name][number]
        for mol_id, algos in model_dict.items():
            for algo_name, masks in algos.items():
                arr = np.array(masks, dtype=float)
                if arr.size == 0:
                    continue
                curr_max = np.max(np.abs(arr))
                key = (model_name_number, algo_name)
                if curr_max > global_max[key]:
                    global_max[key] = curr_max

    # Step 2: accumulate normalized masks and counts
    accumulator = defaultdict(lambda: defaultdict(lambda: None))
    counts = defaultdict(lambda: defaultdict(int))

    for model_name_number in model_names_numbers:
        model_name, number = model_name_number.split("_")
        if model_name not in node_masks or number not in node_masks[model_name]:
            continue
        model_dict = node_masks[model_name][number]

        for mol_id, algos in model_dict.items():
            for algo_name, masks in algos.items():
                arr = np.array(masks, dtype=float)

                # normalize by global max for this model+algo (if non-zero)
                key = (model_name_number, algo_name)
                max_abs = global_max.get(key, 0.0)
                if max_abs > 0:
                    arr = arr / max_abs
                # else leave arr as-is (all zeros or no signal)

                # accumulate
                if accumulator[mol_id][algo_name] is None:
                    accumulator[mol_id][algo_name] = arr.copy()
                else:
                    accumulator[mol_id][algo_name] += arr

                counts[mol_id][algo_name] += 1

    # Step 3: compute mean and convert back to lists
    result = {}
    for mol_id, algos in accumulator.items():
        result[mol_id] = {}
        for algo_name, summed in algos.items():
            c = counts[mol_id][algo_name]
            if c == 0:
                continue
            averaged = (summed / c).tolist()
            result[mol_id][algo_name] = averaged

    return result
