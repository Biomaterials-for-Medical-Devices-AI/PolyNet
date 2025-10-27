from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from polynet.models.GNN import BaseNetwork, BaseNetworkClassifier
from polynet.options.enums import ApplyWeightingToGraph, Networks, Pooling, ProblemTypes


class GCNBase(BaseNetwork):
    def __init__(
        self,
        improved: bool,
        n_node_features: int,
        n_edge_features: int,
        pooling: str = Pooling.GlobalMeanPool,
        n_convolutions: int = 2,
        embedding_dim: int = 64,
        readout_layers: int = 2,
        problem_type: str = ProblemTypes.Regression,
        n_classes: int = 2,
        dropout: float = 0.5,
        cross_att: bool = False,
        apply_weighting_to_graph: str = ApplyWeightingToGraph.BeforePooling,
        seed: int = 42,
    ):
        # Call the constructor of the parent class (BaseNetwork)
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

        # Set class variables
        self._name = Networks.GCN
        self.improved = improved

        # Convolutions
        self.conv_layers = nn.ModuleList([])
        self.conv_layers.append(
            GCNConv(self.n_node_features, self.embedding_dim, improved=self.improved)
        )
        for _ in range(self.n_convolutions - 1):
            self.conv_layers.append(
                GCNConv(self.embedding_dim, self.embedding_dim, improved=self.improved)
            )

        # Batch normalization layers
        self.norm_layers = nn.ModuleList(
            [nn.BatchNorm1d(num_features=self.embedding_dim) for _ in range(self.n_convolutions)]
        )

        if self.cross_att:
            self.monomer_W_att = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.make_readout_layers()

    def get_graph_embedding(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch_index=None,
        edge_attr: Tensor = None,
        monomer_weight: Tensor = None,
    ):
        if (
            monomer_weight is not None
            and self.apply_weighting_to_graph == ApplyWeightingToGraph.BeforeMPP
        ):
            x *= monomer_weight

        for conv_layer, bn in zip(self.conv_layers, self.norm_layers):
            x = F.dropout(
                F.leaky_relu(bn(conv_layer(x, edge_index))), p=self.dropout, training=self.training
            )

        if (
            monomer_weight is not None
            and self.apply_weighting_to_graph == ApplyWeightingToGraph.BeforePooling
        ):
            x *= monomer_weight

        if self.cross_att:
            x = self._cross_attention(x, batch_index, monomer_weight)

        x = self.pooling_fn(x, batch_index)

        return x


class GCNClassifier(GCNBase, BaseNetworkClassifier):
    def __init__(
        self,
        improved: bool,
        n_node_features: int,
        n_edge_features: int,
        pooling: str = Pooling.GlobalMeanPool,
        n_convolutions: int = 2,
        embedding_dim: int = 64,
        readout_layers: int = 2,
        n_classes: int = 2,
        dropout: float = 0.5,
        apply_weighting_to_graph: str = ApplyWeightingToGraph.BeforePooling,
        seed: int = 42,
        cross_att: bool = False,
    ):
        # Call the constructor of the parent class (BaseNetwork)
        super().__init__(
            improved=improved,
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            pooling=pooling,
            n_convolutions=n_convolutions,
            embedding_dim=embedding_dim,
            readout_layers=readout_layers,
            problem_type=ProblemTypes.Classification,
            n_classes=n_classes,
            dropout=dropout,
            cross_att=cross_att,
            apply_weighting_to_graph=apply_weighting_to_graph,
            seed=seed,
        )


class GCNRegressor(GCNBase):
    def __init__(
        self,
        improved: bool,
        n_node_features: int,
        n_edge_features: int,
        pooling: str = Pooling.GlobalMeanPool,
        n_convolutions: int = 2,
        embedding_dim: int = 64,
        readout_layers: int = 2,
        n_classes: int = 1,
        dropout: float = 0.5,
        apply_weighting_to_graph: str = ApplyWeightingToGraph.BeforePooling,
        seed: int = 42,
        cross_att: bool = False,
    ):
        # Call the constructor of the parent class (BaseNetwork)
        super().__init__(
            improved=improved,
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            pooling=pooling,
            n_convolutions=n_convolutions,
            embedding_dim=embedding_dim,
            readout_layers=readout_layers,
            problem_type=ProblemTypes.Regression,
            n_classes=n_classes,
            dropout=dropout,
            cross_att=cross_att,
            apply_weighting_to_graph=apply_weighting_to_graph,
            seed=seed,
        )
