"""
polynet.explainability
=======================
GNN attribution calculation, fragment importance, embedding analysis,
and visualisation for polymer property prediction models.

All computation functions return plain data structures with no Streamlit
dependency. The app layer is responsible for rendering.

::

    from polynet.explainability import run_explanation, ExplanationResult
    from polynet.explainability import get_graph_embeddings, reduce_embeddings
    from polynet.explainability import get_fragment_importance
"""

from polynet.explainability.attributions import (
    build_explainer,
    build_explainers,
    calculate_attributions,
    deep_update,
    get_node_feat_vector_sizes,
    merge_attribution_masks,
    slice_masks_to_feature,
)
from polynet.explainability.embeddings import get_graph_embeddings, reduce_embeddings
from polynet.explainability.fragments import get_fragment_importance
from polynet.explainability.pipeline import ExplanationResult, MoleculeExplanation, run_explanation
from polynet.explainability.visualization import (
    get_cmap,
    plot_attribution_distribution,
    plot_mols_with_numeric_weights,
    plot_mols_with_weights,
    plot_projection_embeddings,
)

__all__ = [
    # Pipeline
    "run_explanation",
    "ExplanationResult",
    "MoleculeExplanation",
    # Attributions
    "build_explainer",
    "build_explainers",
    "calculate_attributions",
    "merge_attribution_masks",
    "slice_masks_to_feature",
    "get_node_feat_vector_sizes",
    "deep_update",
    # Fragments
    "get_fragment_importance",
    # Embeddings
    "get_graph_embeddings",
    "reduce_embeddings",
    # Visualization
    "get_cmap",
    "plot_mols_with_weights",
    "plot_mols_with_numeric_weights",
    "plot_attribution_distribution",
    "plot_projection_embeddings",
]
