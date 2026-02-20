from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from polynet.models.GNN import BaseNetwork, BaseNetworkClassifier
from polynet.options.enums import ApplyWeightingToGraph, Networks, Pooling, ProblemTypes


class GAT(BaseNetwork):

    def __init__(
        self,
        num_heads: int,
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
        self._name = Networks.GAT
        self.num_heads = num_heads

        # Convolutions
        self.conv_layers = nn.ModuleList([])
        self.conv_layers.append(
            GATv2Conv(
                in_channels=self.n_node_features,
                out_channels=self.embedding_dim,
                edge_dim=self.n_edge_features,
                heads=self.num_heads,
                concat=False,
                dropout=self.dropout,
            )
        )
        for _ in range(self.n_convolutions - 1):
            self.conv_layers.append(
                GATv2Conv(
                    in_channels=self.embedding_dim,
                    out_channels=self.embedding_dim,
                    edge_dim=self.n_edge_features,
                    heads=self.num_heads,
                    concat=False,
                    dropout=self.dropout,
                )
            )

        # Batch normalization layers
        self.norm_layers = nn.ModuleList(
            [nn.BatchNorm1d(num_features=self.embedding_dim) for _ in range(self.n_convolutions)]
        )

        if self.cross_att:
            self.monomer_W_att = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.make_readout_layers()


class GATClassifier(GAT, BaseNetworkClassifier):

    def __init__(
        self,
        num_heads: int,
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
            num_heads=num_heads,
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


class GATRegressor(GAT):

    def __init__(
        self,
        num_heads: int,
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
            num_heads=num_heads,
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
