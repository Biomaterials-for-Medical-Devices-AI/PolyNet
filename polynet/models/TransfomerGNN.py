import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

from polynet.models.GNN import BaseNetwork
from polynet.options.enums import Networks, Pooling, ProblemTypes
from torch import Tensor


class TransformerGNN(BaseNetwork):
    def __init__(
        self,
        NumHeads: int,
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
            seed=seed,
        )

        # Set class variables
        self._name = Networks.TransformerGNN
        self.num_heads = NumHeads

        # Convolutions
        self.conv_layers = nn.ModuleList([])
        self.conv_layers.append(
            TransformerConv(
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
                TransformerConv(
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

        # graph embedding is the concatenation of the global mean and max pooling, thus 2*embedding_dim
        graph_embedding = self.graph_embedding

        # Readout layers
        self.readout = nn.ModuleList([])
        for _ in range(self.readout_layers - 1):
            reduced_dim = int(graph_embedding / 2)
            self.readout.append(
                nn.Sequential(nn.Linear(graph_embedding, reduced_dim), nn.BatchNorm1d(reduced_dim))
            )
            graph_embedding = reduced_dim

        # Final readout layer
        self.output_layer = nn.Linear(graph_embedding, self.n_classes)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch_index=None,
        edge_attr: Tensor = None,
        edge_weight=None,
        monomer_weight: Tensor = None,
    ):

        for conv_layer, bn in zip(self.conv_layers, self.norm_layers):
            x = F.dropout(
                F.leaky_relu(bn(conv_layer(x=x, edge_index=edge_index, edge_attr=edge_attr))),
                p=self.dropout,
                training=self.training,
            )

        if monomer_weight is not None:
            x *= monomer_weight

        if self.cross_att:
            x = self._cross_attention(x, batch_index, monomer_weight)

        x = self.pooling_fn(x, batch_index)

        for layer in self.readout:
            x = F.dropout(F.leaky_relu(layer(x)), p=self.dropout, training=self.training)

        x = self.output_layer(x)

        if self.n_classes == 1:
            x = x.float()

        return x

    def return_graph_embedding(
        self, x, edge_index, batch_index=None, edge_attr=None, edge_weight=None, monomer_weight=None
    ):

        for conv_layer, bn in zip(self.conv_layers, self.norm_layers):
            x = F.dropout(
                F.leaky_relu(bn(conv_layer(x=x, edge_index=edge_index, edge_attr=edge_attr))),
                p=self.dropout,
                training=self.training,
            )

        if monomer_weight is not None:
            x *= monomer_weight

        if self.cross_att:
            x = self._cross_attention(x, batch_index, monomer_weight)

        x = self.pooling_fn(x, batch_index)

        return x


class TransformerGNNClassifier(TransformerGNN):
    def __init__(
        self,
        NumHeads: int,
        n_node_features: int,
        n_edge_features: int,
        pooling: str = Pooling.GlobalMeanPool,
        n_convolutions: int = 2,
        embedding_dim: int = 64,
        readout_layers: int = 2,
        n_classes: int = 2,
        dropout: float = 0.5,
        seed: int = 42,
        cross_att: bool = False,
    ):
        # Call the constructor of the parent class (BaseNetwork)
        super().__init__(
            NumHeads=NumHeads,
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
            seed=seed,
        )


class TransformerGNNRegressor(TransformerGNN):
    def __init__(
        self,
        NumHeads: int,
        n_node_features: int,
        n_edge_features: int,
        pooling: str = Pooling.GlobalMeanPool,
        n_convolutions: int = 2,
        embedding_dim: int = 64,
        readout_layers: int = 2,
        n_classes: int = 1,
        dropout: float = 0.5,
        seed: int = 42,
        cross_att: bool = False,
    ):
        # Call the constructor of the parent class (BaseNetwork)
        super().__init__(
            NumHeads=NumHeads,
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
            seed=seed,
        )
