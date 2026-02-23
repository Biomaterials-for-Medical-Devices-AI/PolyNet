"""
polynet.models.gnn.mpnn
========================
Message Passing Neural Network (MPNN) for polymer property prediction.

Uses ``NNConv`` from PyG, where a small edge network (MLP) maps edge
features to a per-edge weight matrix that modulates node message passing.
This makes MPNN one of the most expressive edge-feature-aware architectures
available for molecular graphs.

Like CGGNN, MPNN includes a node projection layer that maps raw node
features to ``embedding_dim`` before the first convolution.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import NNConv

from polynet.config.enums import ApplyWeightingToGraph, Network, Pooling, ProblemType
from polynet.models.base import BaseNetwork, BaseNetworkClassifier


class MPNNBase(BaseNetwork):
    """
    MPNN architecture base — shared by ``MPNNClassifier`` and ``MPNNRegressor``.

    The edge network is an MLP that maps edge features to a
    ``(embedding_dim × embedding_dim)`` weight matrix, which is used to
    compute per-edge messages during aggregation.
    """

    def __init__(
        self,
        n_node_features: int,
        n_edge_features: int,
        pooling: Pooling | str = Pooling.GlobalMeanPool,
        n_convolutions: int = 2,
        embedding_dim: int = 64,
        readout_layers: int = 2,
        problem_type: ProblemType | str = ProblemType.Regression,
        n_classes: int = 1,
        dropout: float = 0.5,
        cross_att: bool = False,
        apply_weighting_to_graph: ApplyWeightingToGraph | str = ApplyWeightingToGraph.BeforePooling,
        seed: int = 42,
    ) -> None:
        super().__init__(
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            pooling=pooling,
            n_convolutions=n_convolutions,
            embedding_dim=embedding_dim,
            readout_layers=readout_layers,
            problem_type=problem_type,
            n_classes=n_classes,
            dropout=dropout,
            cross_att=cross_att,
            apply_weighting_to_graph=apply_weighting_to_graph,
            seed=seed,
        )

        self._name = Network.MPNN

        # Project raw node features → embedding_dim before convolutions
        self.project_nodes = nn.Linear(self.n_node_features, self.embedding_dim)

        self.conv_layers = nn.ModuleList(
            [
                NNConv(
                    in_channels=self.embedding_dim,
                    out_channels=self.embedding_dim,
                    nn=self._make_edge_network(),
                    aggr="add",
                )
                for _ in range(self.n_convolutions)
            ]
        )
        self.norm_layers = nn.ModuleList(
            [nn.BatchNorm1d(self.embedding_dim) for _ in range(self.n_convolutions)]
        )

        if self.cross_att:
            self.monomer_W_att = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.make_readout_layers()

    def _make_edge_network(self) -> nn.Sequential:
        """
        Build the edge MLP that maps edge features to a
        ``(embedding_dim × embedding_dim)`` weight matrix for NNConv.
        """
        return nn.Sequential(
            nn.Linear(self.n_edge_features, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim * self.embedding_dim),
        )

    def get_graph_embedding(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch_index=None,
        edge_attr: Tensor | None = None,
        monomer_weight: Tensor | None = None,
    ) -> Tensor:
        """MPNN-specific embedding — includes node projection before convolutions."""
        if (
            monomer_weight is not None
            and self.apply_weighting_to_graph == ApplyWeightingToGraph.BeforeMPP
        ):
            x = x * monomer_weight

        x = F.leaky_relu(self.project_nodes(x))

        for conv, bn in zip(self.conv_layers, self.norm_layers):
            x = F.dropout(
                F.leaky_relu(bn(conv(x=x, edge_index=edge_index, edge_attr=edge_attr))),
                p=self.dropout,
                training=self.training,
            )

        if (
            monomer_weight is not None
            and self.apply_weighting_to_graph == ApplyWeightingToGraph.BeforePooling
        ):
            x = x * monomer_weight

        if self.cross_att:
            x = self._cross_attention(x, batch_index, monomer_weight)

        return self.pooling_fn(x, batch_index)


class MPNNClassifier(MPNNBase, BaseNetworkClassifier):
    """MPNN model for polymer property classification."""

    def __init__(
        self,
        n_node_features: int,
        n_edge_features: int,
        pooling: Pooling | str = Pooling.GlobalMeanPool,
        n_convolutions: int = 2,
        embedding_dim: int = 64,
        readout_layers: int = 2,
        n_classes: int = 2,
        dropout: float = 0.5,
        cross_att: bool = False,
        apply_weighting_to_graph: ApplyWeightingToGraph | str = ApplyWeightingToGraph.BeforePooling,
        seed: int = 42,
    ) -> None:
        super().__init__(
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            pooling=pooling,
            n_convolutions=n_convolutions,
            embedding_dim=embedding_dim,
            readout_layers=readout_layers,
            problem_type=ProblemType.Classification,
            n_classes=n_classes,
            dropout=dropout,
            cross_att=cross_att,
            apply_weighting_to_graph=apply_weighting_to_graph,
            seed=seed,
        )


class MPNNRegressor(MPNNBase):
    """MPNN model for polymer property regression."""

    def __init__(
        self,
        n_node_features: int,
        n_edge_features: int,
        pooling: Pooling | str = Pooling.GlobalMeanPool,
        n_convolutions: int = 2,
        embedding_dim: int = 64,
        readout_layers: int = 2,
        n_classes: int = 1,
        dropout: float = 0.5,
        cross_att: bool = False,
        apply_weighting_to_graph: ApplyWeightingToGraph | str = ApplyWeightingToGraph.BeforePooling,
        seed: int = 42,
    ) -> None:
        super().__init__(
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            pooling=pooling,
            n_convolutions=n_convolutions,
            embedding_dim=embedding_dim,
            readout_layers=readout_layers,
            problem_type=ProblemType.Regression,
            n_classes=n_classes,
            dropout=dropout,
            cross_att=cross_att,
            apply_weighting_to_graph=apply_weighting_to_graph,
            seed=seed,
        )
