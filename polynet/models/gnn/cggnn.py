"""
polynet.models.gnn.cggnn
========================
Former name of ``polynet.models.gnn.cgcnn``, kept so that GNN models saved
before 1.0.0 (pickled with these class paths) still load.
"""

from polynet.models.gnn.cgcnn import CGCNNBase as CGGNNBase
from polynet.models.gnn.cgcnn import CGCNNClassifier as CGGNNClassifier
from polynet.models.gnn.cgcnn import CGCNNRegressor as CGGNNRegressor

__all__ = ["CGGNNBase", "CGGNNClassifier", "CGGNNRegressor"]
