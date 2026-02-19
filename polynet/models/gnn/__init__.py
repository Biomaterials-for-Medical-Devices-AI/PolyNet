"""
polynet.models.gnn
==================
GNN architecture implementations for polymer property prediction.

Each architecture provides a ``*Classifier`` and ``*Regressor`` variant.
All classes inherit from ``polynet.models.base.BaseNetwork`` or
``polynet.models.base.BaseNetworkClassifier``.

GraphSAGE, MPNN, and TransformerGNN follow the identical pattern
established by GCN, GAT, and CGGNN and should be migrated from
``polynet.models.graphsage``, ``polynet.models.MPNN``, and
``polynet.models.TransformerGNN`` respectively.

::

    from polynet.models.gnn import GCNRegressor, GATClassifier
"""

from polynet.models.gnn.cggnn import CGGNNClassifier, CGGNNRegressor
from polynet.models.gnn.gat import GATClassifier, GATRegressor
from polynet.models.gnn.gcn import GCNClassifier, GCNRegressor
from polynet.models.gnn.graphsage import GraphSAGEClassifier, GraphSAGERegressor
from polynet.models.gnn.mpnn import MPNNClassifier, MPNNRegressor
from polynet.models.gnn.transformer import TransformerGNNClassifier, TransformerGNNRegressor

__all__ = [
    "GCNClassifier",
    "GCNRegressor",
    "GATClassifier",
    "GATRegressor",
    "CGGNNClassifier",
    "CGGNNRegressor",
    "GraphSAGEClassifier",
    "GraphSAGERegressor",
    "MPNNClassifier",
    "MPNNRegressor",
    "TransformerGNNClassifier",
    "TransformerGNNRegressor",
]
