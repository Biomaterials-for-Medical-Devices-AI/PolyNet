"""
polynet.models.gnn.graphsage
=============================
GraphSAGE (Graph Sample and Aggregate) for polymer property prediction.

Like GCN, SAGEConv does not use edge features during message passing.
If your representation includes edge features, consider GAT, TransformerGNN,
or MPNN instead.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SAGEConv

from polynet.config.enums import ApplyWeightingToGraph, Network, Pooling, ProblemType
from polynet.models.base import BaseNetwork, BaseNetworkClassifier


class GraphSAGEBase(BaseNetwork):
    """
    GraphSAGE architecture base — shared by ``GraphSAGEClassifier`` and ``GraphSAGERegressor``.

    Parameters
    ----------
    bias:
        Whether to include a bias term in each SAGEConv layer.
    """

    def __init__(
        self,
        bias: bool,
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

        self._name = Network.GraphSAGE
        self.bias = bias

        self.conv_layers = nn.ModuleList(
            [
                SAGEConv(self.n_node_features, self.embedding_dim, bias=bias),
                *[
                    SAGEConv(self.embedding_dim, self.embedding_dim, bias=bias)
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

    def get_graph_embedding(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch_index=None,
        edge_attr: Tensor | None = None,
        monomer_weight: Tensor | None = None,
    ) -> Tensor:
        """
        GraphSAGE-specific embedding — ``edge_attr`` is intentionally ignored
        as SAGEConv does not support edge features.
        """
        if (
            monomer_weight is not None
            and self.apply_weighting_to_graph == ApplyWeightingToGraph.BeforeMPP
        ):
            x = x * monomer_weight

        for conv, bn in zip(self.conv_layers, self.norm_layers):
            x = F.dropout(
                F.leaky_relu(bn(conv(x, edge_index))), p=self.dropout, training=self.training
            )

        if (
            monomer_weight is not None
            and self.apply_weighting_to_graph == ApplyWeightingToGraph.BeforePooling
        ):
            x = x * monomer_weight

        if self.cross_att:
            x = self._cross_attention(x, batch_index, monomer_weight)

        return self.pooling_fn(x, batch_index)


class GraphSAGEClassifier(GraphSAGEBase, BaseNetworkClassifier):
    """GraphSAGE model for polymer property classification."""

    def __init__(
        self,
        bias: bool,
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
            bias=bias,
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


class GraphSAGERegressor(GraphSAGEBase):
    """GraphSAGE model for polymer property regression."""

    def __init__(
        self,
        bias: bool,
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
            bias=bias,
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
