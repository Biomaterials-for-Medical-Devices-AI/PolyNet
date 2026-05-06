"""
polynet.explainability
=======================
GNN attribution calculation, embedding analysis, and visualisation for
polymer property prediction models.

All computation functions return plain data structures with no Streamlit
dependency. The app layer is responsible for rendering.

High-level masking API (no Streamlit)::

    from polynet.explainability import (
        compute_and_cache_masking,
        build_display_data,
        compute_global_attribution,
        compute_local_attribution,
        GlobalAttributionResult,
        MolAttributionResult,
    )

Low-level building blocks::

    from polynet.explainability import (
        calculate_masking_attributions,
        fragment_attributions_to_distribution,
        merge_fragment_attributions,
        fragment_attributions_to_atom_weights,
    )

Embeddings::

    from polynet.explainability import get_graph_embeddings, reduce_embeddings

TML SHAP API (no Streamlit)::

    from polynet.explainability import (
        compute_global_shap_attribution,
        compute_local_shap_attribution,
        InstanceAttributionResult,
    )
"""

from polynet.explainability.attributions import deep_update
from polynet.explainability.embeddings import get_graph_embeddings, reduce_embeddings
from polynet.explainability.explain import (
    GlobalAttributionResult,
    MolAttributionResult,
    build_display_data,
    compute_and_cache_masking,
    compute_global_attribution,
    compute_local_attribution,
    fragment_attributions_to_atom_weights,
)
from polynet.explainability.masking import (
    calculate_masking_attributions,
    fragment_attributions_to_distribution,
    merge_fragment_attributions,
)
from polynet.explainability.shap_explain import (
    InstanceAttributionResult,
    compute_global_shap_attribution,
    compute_local_shap_attribution,
)
from polynet.explainability.visualization import (
    get_cmap,
    plot_attribution_bar,
    plot_attribution_distribution,
    plot_attribution_strip,
    plot_mols_with_numeric_weights,
    plot_mols_with_weights,
    plot_projection_embeddings,
)

__all__ = [
    # High-level masking pipeline (GNN)
    "compute_and_cache_masking",
    "build_display_data",
    "compute_global_attribution",
    "compute_local_attribution",
    "GlobalAttributionResult",
    "MolAttributionResult",
    "fragment_attributions_to_atom_weights",
    # Attribution cache utility
    "deep_update",
    # Masking attributions (low-level)
    "calculate_masking_attributions",
    "merge_fragment_attributions",
    "fragment_attributions_to_distribution",
    # Embeddings
    "get_graph_embeddings",
    "reduce_embeddings",
    # TML SHAP pipeline
    "compute_global_shap_attribution",
    "compute_local_shap_attribution",
    "InstanceAttributionResult",
    # Visualization
    "get_cmap",
    "plot_mols_with_weights",
    "plot_mols_with_numeric_weights",
    "plot_attribution_distribution",
    "plot_attribution_bar",
    "plot_attribution_strip",
    "plot_projection_embeddings",
]
