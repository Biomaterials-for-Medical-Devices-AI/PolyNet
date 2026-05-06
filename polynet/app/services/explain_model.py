from pathlib import Path

from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
from torch.nn import Module
from torch_geometric.loader import DataLoader

from polynet.config.enums import (
    AttributionPlotType,
    DimensionalityReduction,
    ExplainAlgorithm,
    FragmentationMethod,
    ImportanceNormalisationMethod,
    ProblemType,
)
from polynet.explainability import (
    GlobalAttributionResult,
    MolAttributionResult,
    compute_global_attribution,
    compute_local_attribution,
)
from polynet.explainability.visualization import plot_projection_embeddings
from polynet.featurizer.polymer_graph import CustomPolymerGraph

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
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

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
    """Compute and render the population-level fragment attribution plot."""
    result: GlobalAttributionResult = compute_global_attribution(
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
        plot_type=plot_type,
    )

    st.info(
        f"Distribution over **{result.n_mols}** molecule(s) × **{result.n_models}** model(s) — "
        f"**{result.n_frags_total}** unique fragments found, "
        f"showing top/bottom **{result.n_shown // 2}** each."
        + (f" | class `{result.target_class}`" if result.target_class is not None else "")
        + f" | normalisation: `{result.normalisation_type}`"
    )

    if result.warning:
        st.warning(result.warning)
        return

    st.pyplot(result.figure, use_container_width=True)


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
    """Compute and render per-molecule attribution panels (table + atom heatmap)."""
    mol_results: list[MolAttributionResult] = compute_local_attribution(
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
        mol_names=mol_names,
        predictions=predictions,
        class_labels=class_labels,
    )

    for result in mol_results:
        container = st.container(border=True, key=f"local_mol_{result.mol_idx}_container")
        container.info(result.info_msg)
        container.write(f"True label: `{result.true_label}`")
        container.write(f"Predicted label: `{result.predicted_label}`")

        if result.warning:
            container.warning(result.warning)
            continue

        container.dataframe(result.attribution_df, use_container_width=True)
        container.pyplot(result.mol_figure, use_container_width=True)
