from abc import abstractmethod

import torch
import torch.nn as nn
from torch_geometric.seed import seed_everything

from polynet.options.enums import Pooling, ProblemTypes


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
        self.seed = seed
        self._seed_everything(seed)

        if self.pooling == Pooling.GlobalMeanMaxPool:
            self.graph_embedding = self.embedding_dim * 2
        else:
            self.graph_embedding = self.embedding_dim

    @abstractmethod
    def forward(self):
        raise NotImplementedError

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
