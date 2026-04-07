"""
polynet.pipeline
================
Shared pipeline stage functions used by both the script runner
(``scripts/run_pipeline.py``) and the Streamlit app.

Import from this package for convenience::

    from polynet.pipeline import (
        build_graph_dataset,
        compute_descriptors,
        compute_data_splits,
        train_gnn,
        run_gnn_inference,
        train_tml,
        run_tml_inference,
        compute_metrics,
        plot_results_stage,
        run_explainability,
    )
"""

from polynet.pipeline.stages import (
    build_graph_dataset,
    compute_data_splits,
    compute_descriptors,
    compute_metrics,
    plot_results_stage,
    run_explainability,
    run_gnn_inference,
    run_tml_inference,
    train_gnn,
    train_tml,
)

__all__ = [
    "build_graph_dataset",
    "compute_descriptors",
    "compute_data_splits",
    "train_gnn",
    "run_gnn_inference",
    "train_tml",
    "run_tml_inference",
    "compute_metrics",
    "plot_results_stage",
    "run_explainability",
]
