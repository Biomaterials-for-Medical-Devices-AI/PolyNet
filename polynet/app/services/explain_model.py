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


def explain_model(
    model,
    experiment_path: Path,
    dataset: CustomPolymerGraph,
    explain_mols: list,
    explain_algorithm: ExplainAlgorithms,
    problem_type: ProblemTypes,
    neg_color: str = "#40bcde",
    pos_color: str = "#e64747",
):

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

    custom_cmap = get_cmap(neg_color=neg_color, pos_color=pos_color)

    explain_path = explanation_parent_directory(experiment_path)
    if not explain_path.exists():
        explain_path.mkdir(parents=True, exist_ok=True)

    explanation_file = explanation_json_file_path(experiment_path=experiment_path)

    if explanation_file.exists():
        with open(explanation_file, "r") as f:
            attrs_mol = f.read()

    else:
        attrs_mol = None

    mols = filter_dataset_by_ids(dataset, explain_mols)

    node_masks = {}

    for mol in mols:

        mol_idx = mol.idx

        if attrs_mol is not None and mol_idx in attrs_mol:

            node_mask = attrs_mol[mol_idx]
        else:
            node_mask = explainer(
                x=mol.x,
                edge_index=mol.edge_index,
                batch_index=None,
                edge_attr=mol.edge_attr,
                edge_weight=None,
                monomer_weight=mol.weight_monomer,
            ).node_mask

        if mol_idx not in node_masks:
            node_masks[mol_idx] = {}
        node_masks[mol_idx][explain_algorithm] = node_mask.detach().numpy().tolist()

    for mol in mols:
        masks = np.array(node_masks[mol.idx][explain_algorithm]).sum(axis=1).tolist()
        masks_mol = []
        ia = 0
        for smiles in mol.mols:
            molecule = Chem.MolFromSmiles(smiles)
            n_atoms = molecule.GetNumAtoms()
            st.write(f"Number of atoms in molecule: {n_atoms}")
            masks_mol.append(masks[ia : ia + n_atoms])
            ia += n_atoms
        st.write(masks_mol)
        fig = plot_mols_with_weights(smiles_list=mol.mols, weights_list=masks_mol)
        st.pyplot(fig, use_container_width=True)
