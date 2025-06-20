from abc import abstractmethod

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gmeanp
from torch_geometric.seed import seed_everything

from polynet.options.enums import ApplyWeightingToGraph, Pooling, ProblemTypes


def max_mean_pool(x, batch_index):
    """Concatenates Global Max Pool and Global Mean Pool along feature dimension."""
    return torch.cat([gmp(x, batch_index), gmeanp(x, batch_index)], dim=1)


POOLING_FUNCTIONS = {
    Pooling.GlobalMaxPool: gmp,
    Pooling.GlobalMeanPool: gmeanp,
    Pooling.GlobalAddPool: gap,
    Pooling.GlobalMeanMaxPool: max_mean_pool,
}


class BaseNetwork(nn.Module):
    def __init__(
        self,
        n_node_features: int,
        n_edge_features: int,
        pooling: str = Pooling.GlobalMeanPool,
        n_convolutions: int = 2,
        embedding_dim: int = 64,
        readout_layers: int = 2,
        problem_type: str = ProblemTypes.Regression,
        n_classes: int = 1,
        dropout: float = 0.5,
        cross_att: bool = False,
        apply_weighting_to_graph: str = ApplyWeightingToGraph.BeforePooling,
        seed: int = 42,
    ):
        super().__init__()

        self._name = "BaseNetwork"
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.pooling = pooling
        self.n_convolutions = n_convolutions
        self.embedding_dim = embedding_dim
        self.readout_layers = readout_layers
        self.problem_type = problem_type
        self.n_classes = n_classes
        self.dropout = dropout
        self.cross_att = cross_att
        self.apply_weighting_to_graph = apply_weighting_to_graph
        self.seed = seed
        self._seed_everything(seed)

        self.pooling_fn = POOLING_FUNCTIONS.get(pooling, gmeanp)

        if self.pooling == Pooling.GlobalMeanMaxPool:
            self.graph_embedding = self.embedding_dim * 2
        else:
            self.graph_embedding = self.embedding_dim

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch_index=None,
        edge_attr: Tensor = None,
        monomer_weight: Tensor = None,
    ):
        """
        Forward pass of the model.
        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge indices.
            batch_index (Tensor, optional): Batch indices for graph-level pooling.
            edge_attr (Tensor, optional): Edge attributes.
            monomer_weight (Tensor, optional): Monomer weights for weighting features.
        Returns:
            Tensor: Graph-level embeddings or predictions.
        """

        graph_embedding = self.get_graph_embedding(
            x=x,
            edge_index=edge_index,
            batch_index=batch_index,
            edge_attr=edge_attr,
            monomer_weight=monomer_weight,
        )

        preds = self.readout_function(graph_embedding)

        if self.n_classes == 1:
            preds = preds.float()

        return preds

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
                F.leaky_relu(bn(conv_layer(x=x, edge_index=edge_index, edge_attr=edge_attr))),
                p=self.dropout,
                training=self.training,
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

    def readout_function(self, x: Tensor):
        """
        Applies a series of readout layers to the input tensor.
        Args:
            x (Tensor): Input tensor.
        Returns:
            Tensor: Output tensor after applying readout layers.
        """
        for layer in self.readout:
            x = F.dropout(F.leaky_relu(layer(x)), p=self.dropout, training=self.training)

        x = self.output_layer(x)

        return x

    @property
    def name(self):
        return self._name

    def _seed_everything(self, seed):
        seed_everything(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def reset_parameters(self):
        self._seed_everything(self.seed)
        """Reinitializes all model parameters."""
        for module in self.children():  # Use children() instead of modules()
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def _split_batched_atom_feats(self, node_feats, batch_index):
        """
        Split the batched atom features into a list of atom features
        Args:
            node_feats (torch.Tensor): Batched atom features
            batch_index (torch.Tensor): Batch index tensor
        Returns:
            list: List of atom features
        """
        return [node_feats[batch_index == i] for i in range(batch_index.max().item() + 1)]

    def _split_monomer_feats(self, node_feats, weight_monomer):
        """
        Split the batched atom features into a list of atom features
        Args:
            node_feats (torch.Tensor): Batched atom features
            weight_monomer (torch.Tensor): Monomer weight tensor
        Returns:
            list: List of atom features
        """
        unique_vals = torch.unique(weight_monomer)
        weight_monomer = weight_monomer.squeeze()
        return [node_feats[weight_monomer == i] for i in unique_vals]

    def _cross_attention(self, x, batch_index, monomer_weight):

        split_feats = self._split_batched_atom_feats(x, batch_index)
        split_weights = self._split_batched_atom_feats(monomer_weight, batch_index)

        updated_polymer_feats = []

        for polymer_feats, polymer_weights in zip(split_feats, split_weights):

            monomer_feats = self._split_monomer_feats(polymer_feats, polymer_weights)

            monomer1_pairwise_feats = F.leaky_relu(self.monomer_W_att(monomer_feats[0]))
            monomer2_pairwise_feats = F.leaky_relu(self.monomer_W_att(monomer_feats[1]))

            pairwise_pred = torch.sigmoid(
                torch.matmul(monomer1_pairwise_feats, monomer2_pairwise_feats.t())
            )

            new_monomer2_feats = torch.matmul(pairwise_pred.t(), monomer1_pairwise_feats)
            new_monomer1_feats = torch.matmul(pairwise_pred, monomer2_pairwise_feats)

            new_monomer1_feats += monomer_feats[0]
            new_monomer2_feats += monomer_feats[1]

            new_polymer_feats = torch.cat([new_monomer1_feats, new_monomer2_feats], dim=0)

            updated_polymer_feats.append(new_polymer_feats)

        x = torch.cat(updated_polymer_feats, dim=0)

        return x

    def predict(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch_index=None,
        edge_attr: Tensor = None,
        monomer_weight: Tensor = None,
    ):
        """
        Predicts the output for the given input.
        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge indices.
            batch_index (Tensor, optional): Batch indices for graph-level pooling.
            edge_attr (Tensor, optional): Edge attributes.
            monomer_weight (Tensor, optional): Monomer weights for weighting features.
        Returns:
            Tensor: Predicted output.
        """

        preds = (
            self.forward(
                x=x,
                edge_index=edge_index,
                batch_index=batch_index,
                edge_attr=edge_attr,
                monomer_weight=monomer_weight,
            )
            .detach()
            .numpy()
            .flatten()
        )

        return preds


class BaseNetworkClassifier(BaseNetwork):
    def __init__(
        self,
        n_node_features: int,
        n_edge_features: int,
        pooling: str = Pooling.GlobalMeanPool,
        n_convolutions: int = 2,
        embedding_dim: int = 64,
        readout_layers: int = 2,
        problem_type: str = ProblemTypes.Classification,
        n_classes: int = 1,
        dropout: float = 0.5,
        cross_att: bool = False,
        apply_weighting_to_graph: str = ApplyWeightingToGraph.BeforePooling,
        seed: int = 42,
    ):
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

        self._name = "BaseNetworkClassifier"

    def predict(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch_index=None,
        edge_attr: Tensor = None,
        monomer_weight: Tensor = None,
    ):
        """
        Predicts the output for the given input.
        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge indices.
            batch_index (Tensor, optional): Batch indices for graph-level pooling.
            edge_attr (Tensor, optional): Edge attributes.
            monomer_weight (Tensor, optional): Monomer weights for weighting features.
        Returns:
            Tensor: Predicted class probabilities.
        """
        probs = self.predict_probs(
            x=x,
            edge_index=edge_index,
            batch_index=batch_index,
            edge_attr=edge_attr,
            monomer_weight=monomer_weight,
        )

        preds = np.argmax(probs, axis=1)

        return preds

    def predict_probs(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch_index=None,
        edge_attr: Tensor = None,
        monomer_weight: Tensor = None,
    ):
        """
        Predicts the class probabilities for the given input.
        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge indices.
            batch_index (Tensor, optional): Batch indices for graph-level pooling.
            edge_attr (Tensor, optional): Edge attributes.
            monomer_weight (Tensor, optional): Monomer weights for weighting features.
        Returns:
            Tensor: Predicted class probabilities.
        """
        out = self.forward(
            x=x,
            edge_index=edge_index,
            batch_index=batch_index,
            edge_attr=edge_attr,
            monomer_weight=monomer_weight,
        )

        probs = torch.softmax(out, dim=1).detach().numpy()

        return probs
