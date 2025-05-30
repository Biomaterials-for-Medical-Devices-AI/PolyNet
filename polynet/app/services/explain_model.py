import json
from pathlib import Path

import captum
import matplotlib.colors as mcolors
import numpy as np
from rdkit import Chem
import streamlit as st
from torch_geometric.explain import CaptumExplainer, Explainer, GNNExplainer, ModelConfig

from polynet.app.options.file_paths import (
    explanation_json_file_path,
    explanation_parent_directory,
    explanation_plots_path,
)
from polynet.app.utils import filter_dataset_by_ids
from polynet.explain.explain_mol import plot_mols_with_weights
from polynet.featurizer.graph_representation.polymer import CustomPolymerGraph
from polynet.options.enums import ExplainAlgorithms, ProblemTypes

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
    explain_algorithm: ExplainAlgorithms,
    problem_type: ProblemTypes,
    neg_color: str = "#40bcde",
    pos_color: str = "#e64747",
    normalisation_type: str = "local",
    cutoff_explain: float = 0.1,
    mol_names=dict,
):

    cmap = get_cmap(neg_color=neg_color, pos_color=pos_color)

    if problem_type == ProblemTypes.Classification:
        task = "multiclass_classification"
    elif problem_type == ProblemTypes.Regression:
        task = "regression"

    model_config = ModelConfig(mode=task, task_level="graph", return_type="raw")

    if explain_algorithm == ExplainAlgorithms.GNNExplainer:
        algorithm = GNNExplainer(model=model, epochs=100, return_type="raw", explain_graph=True)
    elif explain_algorithm == ExplainAlgorithms.ShapleyValueSampling:
        algorithm = CaptumExplainer(attribution_method=captum.attr.InputXGradient)

    explainer = Explainer(
        model=model,
        algorithm=algorithm,
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type=None,
        model_config=model_config,
    )

    explain_path = explanation_parent_directory(experiment_path)

    if not explain_path.exists():
        explain_path.mkdir(parents=True, exist_ok=True)

    explanation_file = explanation_json_file_path(experiment_path=experiment_path)

    if explanation_file.exists():
        with open(explanation_file) as f:
            attrs_mol = json.load(f)

    else:
        attrs_mol = None

    if isinstance(explain_mols, str):
        explain_mols = [explain_mols]

    mols = filter_dataset_by_ids(dataset, explain_mols)

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
                    edge_weight=None,
                    monomer_weight=mol.weight_monomer,
                )
                .node_mask.detach()
                .numpy()
                .tolist()
            )

        if mol_idx not in node_masks:
            node_masks[mol_idx] = {}
        node_masks[mol_idx][explain_algorithm] = node_mask

    with open(explanation_file, "w") as f:
        json.dump(node_masks, f, indent=4)

    if normalisation_type == "global":

        max_val = None

        for mol in mols:
            maks = np.array(node_masks[mol.idx][explain_algorithm]).sum(axis=1)
            local_max_val = np.max(np.abs(maks))
            if max_val is None or local_max_val > max_val:
                max_val = local_max_val
                st.write(
                    f"New global max value found: {max_val} for molecule {mol.idx} with algorithm {explain_algorithm}"
                )

    for mol in mols:
        names = mol_names.get(mol.idx, None)
        masks = np.array(node_masks[mol.idx][explain_algorithm]).sum(axis=1)

        if normalisation_type == "local":
            masks = masks / np.max(np.abs(masks))
        elif normalisation_type == "global":
            masks = masks / max_val

        masks = np.where(np.abs(masks) > cutoff_explain, masks, 0.0)
        masks = masks.tolist()

        masks_mol = []
        ia = 0
        for smiles in mol.mols:
            molecule = Chem.MolFromSmiles(smiles)
            n_atoms = molecule.GetNumAtoms()
            masks_mol.append(masks[ia : ia + n_atoms])
            ia += n_atoms

        fig = plot_mols_with_weights(
            smiles_list=mol.mols,
            weights_list=masks_mol,
            colormap=cmap,
            legend=names,
            # min_weight=-1.0,
            # max_weight=1.0,
        )
        st.pyplot(fig, use_container_width=True)


def find_highest_contribution(
    masks: dict,
    normalisation_type="local",
    explain_algorithm=ExplainAlgorithms.ShapleyValueSampling,
):
    """
    Normalize the attributions based on the specified normalisation type.
    Args:
        masks (dict): Dictionary containing the attributions for each molecule.
        normalisation_type (str): Type of normalization to apply. Options are "local" or "global".
        cutoff_explain (float): Cutoff value to filter out low-attribution features.
    Returns:
        dict: Normalized attributions.
    """

    if normalisation_type == "local":
        for mol_idx, mol_masks in masks.items():
            for algorithm, mask in mol_masks.items():
                if algorithm != explain_algorithm:
                    continue
                max_val = np.max(np.abs(mask))

    elif normalisation_type == "global":
        all_masks = []
        for mol_masks in masks.values():
            for algorithm, mask in mol_masks.items():
                if algorithm != explain_algorithm:
                    continue
                all_masks.append(mask)
        all_masks = np.concatenate([np.array(mol_masks) for mol_masks in all_masks])
        max_val = np.max(np.abs(all_masks))

    else:
        raise ValueError(f"Unknown normalisation type: {normalisation_type}")

    return max_val
