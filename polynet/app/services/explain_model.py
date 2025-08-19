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
from torch_geometric.explain import CaptumExplainer, Explainer, GNNExplainer, ModelConfig
from torch_geometric.loader import DataLoader

from polynet.app.options.file_paths import (
    explanation_json_file_path,
    explanation_parent_directory,
    explanation_plots_path,
)
from polynet.app.utils import filter_dataset_by_ids
from polynet.explain.explain_mol import (
    get_fragment_importance,
    plot_attribution_distribution,
    plot_mols_with_numeric_weights,
    plot_mols_with_weights,
)
from polynet.featurizer.graph_representation.polymer import CustomPolymerGraph
from polynet.options.enums import (
    AtomBondDescriptorDictKeys,
    DimensionalityReduction,
    ExplainAlgorithms,
    ImportanceNormalisationMethods,
    ProblemTypes,
    Results,
)

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
    reduced_embeddings = reduced_embeddings.to_numpy()

    labels = labels.loc[mols_to_plot]

    projection_fig = plot_projection_embeddings(
        reduced_embeddings, labels=labels, cmap=colormap, style=style_by
    )
    st.pyplot(projection_fig, use_container_width=True)


def explain_model(
    model,
    model_number: int,
    experiment_path: Path,
    dataset: CustomPolymerGraph,
    explain_mols: list,
    plot_mols: list,
    explain_algorithm: ExplainAlgorithms,
    problem_type: ProblemTypes,
    neg_color: str = "#40bcde",
    pos_color: str = "#e64747",
    normalisation_type: str = "local",
    cutoff_explain: float = 0.1,
    mol_names: dict = {},
    predictions: dict = {},
    node_features: dict = {},
    explain_feature: str = "All Features",
):

    # get colormap for visualization
    cmap = get_cmap(neg_color=neg_color, pos_color=pos_color)

    # Set the problem type passed to the model config
    if problem_type == ProblemTypes.Classification:
        task = "multiclass_classification"
    elif problem_type == ProblemTypes.Regression:
        task = "regression"
    # Create the model configuration for explainer
    model_config = ModelConfig(mode=task, task_level="graph", return_type="raw")

    # Initialize the explainer based on the selected algorithm
    if explain_algorithm == ExplainAlgorithms.GNNExplainer:
        algorithm = GNNExplainer(model=model, epochs=100, return_type="raw", explain_graph=True)
    elif explain_algorithm == ExplainAlgorithms.ShapleyValueSampling:
        algorithm = CaptumExplainer(attribution_method=captum.attr.ShapleyValueSampling)
    elif explain_algorithm == ExplainAlgorithms.InputXGradients:
        algorithm = CaptumExplainer(attribution_method=captum.attr.InputXGradient)
    elif explain_algorithm == ExplainAlgorithms.Saliency:
        algorithm = CaptumExplainer(attribution_method=captum.attr.Saliency)
    elif explain_algorithm == ExplainAlgorithms.IntegratedGradients:
        algorithm = CaptumExplainer(attribution_method=captum.attr.IntegratedGradients)
    elif explain_algorithm == ExplainAlgorithms.Deconvolution:
        algorithm = CaptumExplainer(attribution_method=captum.attr.Deconvolution)
    elif explain_algorithm == ExplainAlgorithms.GuidedBackprop:
        algorithm = CaptumExplainer(attribution_method=captum.attr.GuidedBackprop)
    else:
        st.error(f"Unknown explain algorithm: {explain_algorithm}")
        return

    # Initialize the explainer with the model and algorithm
    explainer = Explainer(
        model=model,
        algorithm=algorithm,
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type=None,
        model_config=model_config,
    )

    # Create the directory for explanations if it does not exist
    explain_path = explanation_parent_directory(experiment_path)
    if not explain_path.exists():
        explain_path.mkdir(parents=True, exist_ok=True)

    # Load existing explanations if available
    explanation_file = explanation_json_file_path(experiment_path=experiment_path)
    if explanation_file.exists():
        with open(explanation_file) as f:
            attrs_mol = json.load(f)
    else:
        attrs_mol = None

    if isinstance(explain_mols, str):
        explain_mols = [explain_mols]

    # Filter the dataset to only include the molecules we want to explain
    mols = filter_dataset_by_ids(dataset, explain_mols)

    # Calculate the node masks for the selected molecules
    node_masks = calculate_attributions(
        mols=mols,
        attrs_mol=attrs_mol,
        explain_algorithm=explain_algorithm,
        explainer=explainer,
        model_name=model._name,
        model_number=int(model_number),
    )

    # Save the node masks to a JSON file
    with open(explanation_file, "w") as f:
        json.dump(node_masks, f, indent=4)

    node_masks = node_masks[model._name][model_number]

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

    if normalisation_type == ImportanceNormalisationMethods.Global:

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
        mols=mols, node_masks=node_masks, explain_algorithm=explain_algorithm
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

        if normalisation_type == ImportanceNormalisationMethods.Local:
            masks = masks / np.max(np.abs(masks))
        elif normalisation_type == ImportanceNormalisationMethods.Global:
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

        container.write(f"Plotting molecule {mol.idx} with algorithm {explain_algorithm}")

        container.write(f"True label: `{predictions.get(mol.idx, {}).get(Results.Label, 'N/A')}`")
        container.write(
            f"Predicted label: `{predictions.get(mol.idx, {}).get(Results.Predicted, 'N/A')}`"
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
    attrs_mol: dict,
    explain_algorithm: ExplainAlgorithms,
    explainer: Explainer,
    model_name: str,
    model_number: int,
):
    node_masks = {}

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
                )
                .node_mask.detach()
                .numpy()
                .tolist()
            )

        # Insert in nested structure
        node_masks.setdefault(model_name, {}).setdefault(model_number, {}).setdefault(mol_idx, {})[
            explain_algorithm
        ] = node_mask

    return node_masks


def get_node_feat_vector_size(node_features: dict) -> dict:

    lengths_dict = {}

    for key, value in node_features.items():
        if value == {}:
            lengths_dict[key] = 1
            continue
        allowed_features_size = len(value[AtomBondDescriptorDictKeys.AllowableVals])
        wildcard_feat_size = int(value[AtomBondDescriptorDictKeys.Wildcard])
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
    tsne_embeddings: np.ndarray, labels: list = None, cmap: str = "blues", style: list = None
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

    plt.title("Projection of Graph Embeddings", fontsize=25)
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
        fig.colorbar(sm, ax=ax, label="Value")  # Explicitly pass `ax`
        plt.gca().get_legend().remove()

    return fig
