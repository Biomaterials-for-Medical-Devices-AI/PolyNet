"""
polynet.models.gnn.gat
=======================
Graph Attention Network v2 (GATv2) for polymer property prediction.

Uses ``GATv2Conv`` from PyG which fixes the static attention problem
of the original GAT by computing attention weights after concatenating
query and key features.
"""

from __future__ import annotations

import torch.nn as nn
from torch_geometric.nn import GATv2Conv

from polynet.config.enums import ApplyWeightingToGraph, Network, Pooling, ProblemType
from polynet.models.base import BaseNetwork, BaseNetworkClassifier


class GATBase(BaseNetwork):
    """
    GATv2 architecture base — shared by ``GATClassifier`` and ``GATRegressor``.

    Parameters
    ----------
    num_heads:
        Number of attention heads per GATv2Conv layer.
    """

    def __init__(
        self,
        num_heads: int,
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

        self._name = Network.GAT
        self.num_heads = num_heads

        self.conv_layers = nn.ModuleList(
            [
                GATv2Conv(
                    in_channels=self.n_node_features,
                    out_channels=self.embedding_dim,
                    edge_dim=self.n_edge_features,
                    heads=self.num_heads,
                    concat=False,
                    dropout=self.dropout,
                ),
                *[
                    GATv2Conv(
                        in_channels=self.embedding_dim,
                        out_channels=self.embedding_dim,
                        edge_dim=self.n_edge_features,
                        heads=self.num_heads,
                        concat=False,
                        dropout=self.dropout,
                    )
                    for _ in range(self.n_convolutions - 1)
                ],
            ]
        )
        self.norm_layers = nn.ModuleList(
            [nn.BatchNorm1d(self.embedding_dim) for _ in range(self.n_convolutions)]
        )

        if self.cross_att:
            self.monomer_W_att = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.make_readout_layers()


class GATClassifier(GATBase, BaseNetworkClassifier):
    """GATv2 model for polymer property classification."""

    def __init__(
        self,
        num_heads: int,
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
            num_heads=num_heads,
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


class GATRegressor(GATBase):
    """GATv2 model for polymer property regression."""

    def __init__(
        self,
        num_heads: int,
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
            num_heads=num_heads,
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
