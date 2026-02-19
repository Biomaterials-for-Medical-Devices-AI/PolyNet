"""
polynet.models
==============
GNN model classes for polymer property prediction.

All models share a common interface defined in ``polynet.models.base``:
- ``forward()`` — full forward pass
- ``predict()`` — numpy predictions for a single batch
- ``predict_loader()`` — inference over a full DataLoader
- ``reset_parameters()`` — reinitialise weights with the original seed

Typical usage::

    from polynet.models import GCNRegressor, GATClassifier
    from polynet.config.enums import Pooling

    model = GCNRegressor(
        improved=True,
        n_node_features=32,
        n_edge_features=5,
        embedding_dim=64,
        n_convolutions=3,
        pooling=Pooling.GlobalMeanPool,
        dropout=0.05,
    )
"""

from polynet.models.base import BaseNetwork, BaseNetworkClassifier
from polynet.models.gnn import (
    CGGNNClassifier,
    CGGNNRegressor,
    GATClassifier,
    GATRegressor,
    GCNClassifier,
    GCNRegressor,
    GraphSAGEClassifier,
    GraphSAGERegressor,
    MPNNClassifier,
    MPNNRegressor,
    TransformerGNNClassifier,
    TransformerGNNRegressor,
)

__all__ = [
    # Base classes
    "BaseNetwork",
    "BaseNetworkClassifier",
    # GCN
    "GCNClassifier",
    "GCNRegressor",
    # GAT
    "GATClassifier",
    "GATRegressor",
    # CGGNN
    "CGGNNClassifier",
    "CGGNNRegressor",
    # GraphSAGE
    "GraphSAGEClassifier",
    "GraphSAGERegressor",
    # MPNN
    "MPNNClassifier",
    "MPNNRegressor",
    # TransformerGNN
    "TransformerGNNClassifier",
    "TransformerGNNRegressor",
]
