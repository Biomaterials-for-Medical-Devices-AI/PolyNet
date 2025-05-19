import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_add_pool as gsp
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap

from polynet.models.GNN import BaseNetwork
from polynet.options.enums import Networks, Pooling, ProblemTypes


class GraphSAGE(BaseNetwork):
    def __init__(
        self,
        aggr,
        n_node_features: int,
        n_edge_features: int,
        pooling: str = Pooling.GlobalMeanPool,
        n_convolutions: int = 2,
        embedding_dim: int = 64,
        readout_layers: int = 2,
        problem_type: str = ProblemTypes.Regression,
        n_classes: int = 1,
        dropout: float = 0.5,
        seed: int = 42,
    ):
        # Call the constructor of the parent class (BaseNetwork)
        super().__init__(
            n_node_features,
            n_edge_features,
            pooling,
            n_convolutions,
            embedding_dim,
            readout_layers,
            problem_type,
            n_classes,
            dropout,
            seed,
        )

        # Set class variables
        self._name = Networks.GCN
        self.aggr = aggr

        # Set network architecture

        # First convolution and activation function
        self.linear = nn.Linear(self.n_node_features, self.embedding_dim)
        self.batch_norm = nn.BatchNorm1d(self.embedding_dim)

        # Convolutions
        self.conv_layers = nn.ModuleList([])
        self.norm_layers = nn.ModuleList([])

        for _ in range(self.n_convolutions):
            self.conv_layers.append(
                SAGEConv(self.embedding_dim, self.embedding_dim, aggr=self.aggr)
            )
            self.norm_layers.append(nn.BatchNorm1d(self.embedding_dim))

        # graph embedding is the concatenation of the global mean and max pooling, thus 2*embedding_dim
        graph_embedding = self.graph_embedding

        self.dropout_layer = nn.Dropout(p=self.dropout)

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
        self, x, edge_index, batch_index=None, edge_attr=None, edge_weight=None, monomer_weight=None
    ):
        x = F.leaky_relu(self.batch_norm(self.linear(x)))

        for conv_layer, norm_layer in zip(self.conv_layers, self.norm_layers):
            x = F.leaky_relu(norm_layer(conv_layer(x, edge_index)))

        if monomer_weight is not None:
            x *= monomer_weight

        if self.pooling == Pooling.GlobalMeanPool:
            x = gap(x, batch_index)
        elif self.pooling == Pooling.GlobalMaxPool:
            x = gmp(x, batch_index)
        elif self.pooling == Pooling.GlobalAddPool:
            x = gsp(x, batch_index)
        elif self.pooling == Pooling.GlobalMeanMaxPool:
            x = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        x = self.dropout_layer(x)

        for layer in self.readout:
            x = F.leaky_relu(layer(x))

        x = self.output_layer(x)

        if self.n_classes == 1:
            x = x.float()

        return x
