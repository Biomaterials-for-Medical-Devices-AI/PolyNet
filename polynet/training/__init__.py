"""
polynet.training
================
Training, evaluation, and hyperparameter optimisation for polynet models.

::

    from polynet.training import train_gnn_ensemble, train_tml_ensemble
    from polynet.training import calculate_metrics, compute_class_weights
    from polynet.training import plot_results, plot_learning_curves
"""

from polynet.training.evaluate import plot_learning_curves, plot_results
from polynet.training.gnn import eval_network, train_gnn_ensemble, train_model, train_network
from polynet.training.metrics import calculate_metrics, compute_class_weights, get_metrics
from polynet.training.tml import generate_models, train_tml_ensemble

__all__ = [
    # GNN training
    "train_gnn_ensemble",
    "train_model",
    "train_network",
    "eval_network",
    # TML training
    "train_tml_ensemble",
    "generate_models",
    # Metrics
    "calculate_metrics",
    "get_metrics",
    "compute_class_weights",
    # Evaluation plots
    "plot_results",
    "plot_learning_curves",
]
