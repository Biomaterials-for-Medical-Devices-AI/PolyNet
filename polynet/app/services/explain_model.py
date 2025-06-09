import json
from pathlib import Path

import captum
import matplotlib.colors as mcolors
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS
import streamlit as st
from torch_geometric.explain import CaptumExplainer, Explainer, GNNExplainer, ModelConfig

from polynet.app.options.file_paths import (
    explanation_json_file_path,
    explanation_parent_directory,
    explanation_plots_path,
)
from polynet.app.utils import filter_dataset_by_ids
from polynet.explain.explain_mol import plot_mols_with_weights, plot_mols_with_numeric_weights
from polynet.featurizer.graph_representation.polymer import CustomPolymerGraph
from polynet.options.enums import (
    ExplainAlgorithms,
    ProblemTypes,
    Results,
    AtomBondDescriptorDictKeys,
    AtomFeatures,
)

from scipy.stats import gaussian_kde
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt

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


def explain_model(
    model,
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
        algorithm = CaptumExplainer(attribution_method=captum.attr.InputXGradient)
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
        mols=mols, attrs_mol=attrs_mol, explain_algorithm=explain_algorithm, explainer=explainer
    )

    # Save the node masks to a JSON file
    with open(explanation_file, "w") as f:
        json.dump(node_masks, f, indent=4)

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

    if normalisation_type == "global":

        max_val = None

        for mol in mols:
            maks = node_masks[mol.idx][explain_algorithm].sum(axis=1)
            local_max_val = np.max(np.abs(maks))
            if max_val is None or local_max_val > max_val:
                max_val = local_max_val
                st.write(
                    f"New global max value found: {max_val:.3f} for molecule {mol.idx} with algorithm {explain_algorithm}"
                )

    frags_importances = get_fragment_importance(mols, node_masks, explain_algorithm)

    # st.write(frags_importances)
    # Save the fragment importances to a JSON file

    fig = plot_attribution_distribution(
        attribution_dict=frags_importances, neg_color=neg_color, pos_color=pos_color
    )
    st.pyplot(fig, use_container_width=True)

    plot_mols = filter_dataset_by_ids(dataset, plot_mols)

    for mol in plot_mols:
        container = st.container(border=True, key=f"mol_{mol.idx}_container")
        names = mol_names.get(mol.idx, None)
        masks = node_masks[mol.idx][explain_algorithm].sum(axis=1)

        if normalisation_type == "local":
            masks = masks / np.max(np.abs(masks))
        elif normalisation_type == "global":
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
    mols: list, attrs_mol: dict, explain_algorithm: ExplainAlgorithms, explainer: Explainer
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

        if mol_idx not in node_masks:
            node_masks[mol_idx] = {}
        node_masks[mol_idx][explain_algorithm] = node_mask

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


def plot_attribution_distribution(
    attribution_dict,
    figsize=(8, 10),
    neg_color="#1f77b4",
    pos_color="#ff7f0e",
    kde_bandwidth=0.2,
    linewidth=0.7,
    show_averages=True,
):
    """
    Plot attribution distribution with KDE for multiple values per functional group.

    Parameters:
    -----------
    attribution_dict : dict
        Keys are functional groups (str), values are lists of attribution values
    figsize : tuple, optional
        Figure size (width, height)
    neg_color : str, optional
        Color for negative attributions (blue default)
    pos_color : str, optional
        Color for positive attributions (orange default)
    kde_bandwidth : float, optional
        Bandwidth for KDE smoothing (default 0.2)
    linewidth : float, optional
        Width of distribution lines (default 0.7)
    show_averages : bool, optional
        Whether to show average markers (default True)

    Returns:
    --------
    matplotlib.figure.Figure
    """

    # Prepare data
    labels = list(attribution_dict.keys())
    values = list(attribution_dict.values())

    # Create figure with adjusted layout
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.3)  # Make room for y-axis labels

    # Calculate positions for each label
    y_pos = np.arange(len(labels))

    # Plot zero line
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-", zorder=1)

    # Plot each distribution
    verts_pos = []
    verts_neg = []

    for i, (label, vals) in enumerate(zip(labels, values)):
        vals = np.array(vals)

        # Skip if no data
        if len(vals) == 0:
            continue

        # Add small noise if all values are identical
        if len(vals) > 1 and np.allclose(vals, vals[0]):
            vals = vals + np.random.normal(0, 1e-6, size=len(vals))

        # Calculate average
        avg = np.mean(vals)

        # Create KDE for positive values
        pos_vals = vals[vals > 0]
        if len(pos_vals) > 1:
            try:
                kde_pos = gaussian_kde(pos_vals, bw_method=kde_bandwidth)
                x_pos = np.linspace(0, max(3, vals.max() * 1.5), 100)
                density_pos = kde_pos(x_pos)
                # Scale density for visualization
                density_pos = density_pos / (density_pos.max() + 1e-8) * 0.4
                # Create vertices for polygon
                verts_pos.append(list(zip(x_pos, i + density_pos)) + [(0, i)])
            except np.linalg.LinAlgError:
                pass  # Skip if KDE fails

        # Create KDE for negative values
        neg_vals = vals[vals < 0]
        if len(neg_vals) > 1:
            try:
                kde_neg = gaussian_kde(neg_vals, bw_method=kde_bandwidth)
                x_neg = np.linspace(min(-3, vals.min() * 1.5), 0, 100)
                density_neg = kde_neg(x_neg)
                # Scale density for visualization
                density_neg = density_neg / (density_neg.max() + 1e-8) * 0.4
                # Create vertices for polygon
                verts_neg.append(list(zip(x_neg, i - density_neg)) + [(0, i)])
            except np.linalg.LinAlgError:
                pass  # Skip if KDE fails

    # Plot positive distributions (orange)
    if verts_pos:
        poly_pos = PolyCollection(
            verts_pos,
            facecolors=pos_color,
            edgecolors=pos_color,
            linewidths=linewidth,
            alpha=0.7,
            zorder=2,
        )
        ax.add_collection(poly_pos)

    # Plot negative distributions (blue)
    if verts_neg:
        poly_neg = PolyCollection(
            verts_neg,
            facecolors=neg_color,
            edgecolors=neg_color,
            linewidths=linewidth,
            alpha=0.7,
            zorder=3,
        )
        ax.add_collection(poly_neg)

    # Add average markers if requested
    if show_averages:
        for i, vals in enumerate(values):
            if len(vals) > 0:
                avg = np.mean(np.array(vals))  # Ensure we use original values for average
                color = pos_color if avg > 0 else neg_color
                ax.plot(avg, i, "o", color=color, markersize=8, markeredgecolor="white", zorder=4)
                # Add average value text
                ax.text(
                    avg + 0.1 * np.sign(avg),
                    i,
                    f"{avg:.2f}",
                    va="center",
                    ha="left" if avg > 0 else "right",
                    fontsize=9,
                    zorder=5,
                )

    # Customize axes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Attribution", fontsize=12)
    ax.set_title("Attribution Distribution", fontsize=14, pad=20)

    # Set symmetrical x-limits if needed
    xlim = max(3, np.abs(ax.get_xlim()).max())
    ax.set_xlim(-xlim, xlim)

    # Add grid lines
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    # Custom spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    return fig


def get_fragment_importance(mols, node_masks, explain_algorithm):

    frags_importances = {}

    for mol in mols:
        mol_idx = mol.idx
        frag_importance = node_masks[mol_idx][explain_algorithm]

        break_early = not mol_idx.endswith(")")

        # st.write(frag_importance)

        num_atoms = 0

        for smiles in mol.mols:

            frags = fragment_and_match(smiles)

            for frag_smiles, atom_indices in frags.items():

                if len(frag_smiles) < 3:
                    continue

                if frag_smiles not in frags_importances:
                    frags_importances[frag_smiles] = []

                for idx in atom_indices:
                    idx = [num_atoms + i for i in idx]
                    frag_score = np.sum(frag_importance[idx])
                    frags_importances[frag_smiles].append(frag_score)

            if break_early:
                continue

            num_atoms += Chem.MolFromSmiles(smiles).GetNumAtoms()

    return frags_importances


def sanitize_fragment(frag):
    """Remove dummy atoms from a BRICS fragment to make it matchable."""
    editable = Chem.EditableMol(frag)
    atoms_to_remove = [atom.GetIdx() for atom in frag.GetAtoms() if atom.GetAtomicNum() == 0]

    # Remove from the end to preserve indices
    for idx in sorted(atoms_to_remove, reverse=True):
        editable.RemoveAtom(idx)
    return editable.GetMol()


def fragment_and_match(smiles):
    """
    BRICS-decomposes a molecule and matches cleaned fragments back to the original.

    Parameters:
        smiles (str): SMILES string of the molecule.

    Returns:
        dict: {fragment_smiles: [list of atom index lists in original mol]}
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES provided.")

    # Get BRICS fragments with dummy atoms
    raw_frags = list(BRICS.BRICSDecompose(mol, returnMols=True))

    matches = {}
    for raw_frag in raw_frags:
        frag = sanitize_fragment(raw_frag)
        Chem.SanitizeMol(frag)

        substruct_matches = mol.GetSubstructMatches(frag, uniquify=True)
        if substruct_matches:
            frag_smiles = Chem.MolToSmiles(frag)
            matches[frag_smiles] = [list(match) for match in substruct_matches]

    return matches
