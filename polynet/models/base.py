"""
polynet.models.base
====================
Base GNN model classes shared across all polymer GNN architectures.

``BaseNetwork`` implements the full forward pass, graph embedding extraction,
pooling, readout layers, and prediction interface for regression.
``BaseNetworkClassifier`` extends it with softmax prediction and
class probability output for classification tasks.

All architecture-specific models in ``polynet.models.gnn`` inherit from
one of these two base classes.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool as gap
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gmeanp
from torch_geometric.seed import seed_everything

from polynet.config.enums import ApplyWeightingToGraph, Pooling, ProblemType

# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------


def _max_mean_pool(x: Tensor, batch_index: Tensor) -> Tensor:
    """Concatenate global max pool and global mean pool along the feature dim."""
    return torch.cat([gmp(x, batch_index), gmeanp(x, batch_index)], dim=1)


POOLING_FUNCTIONS: dict = {
    Pooling.GlobalMaxPool: gmp,
    Pooling.GlobalMeanPool: gmeanp,
    Pooling.GlobalAddPool: gap,
    Pooling.GlobalMeanMaxPool: _max_mean_pool,
}


# ---------------------------------------------------------------------------
# Base regression network
# ---------------------------------------------------------------------------


class BaseNetwork(nn.Module):
    """
    Base GNN model for polymer property regression.

    Implements the shared forward pass, graph embedding extraction,
    monomer weighting, cross-attention, pooling, and readout layers.
    Subclasses define ``conv_layers`` and ``norm_layers`` in their
    ``__init__`` and may override ``get_graph_embedding`` when their
    message passing differs from the default.

    Parameters
    ----------
    n_node_features:
        Dimensionality of input node feature vectors.
    n_edge_features:
        Dimensionality of input edge feature vectors.
    pooling:
        Graph-level pooling strategy.
    n_convolutions:
        Number of graph convolutional layers.
    embedding_dim:
        Hidden dimensionality of node embeddings after convolution.
    readout_layers:
        Number of MLP layers in the readout network.
    problem_type:
        Regression or classification. Controls output layer behaviour.
    n_classes:
        Number of output units. Use ``1`` for regression, ``n`` for
        n-class classification.
    dropout:
        Dropout probability applied after each conv and readout layer.
    cross_att:
        Whether to apply cross-monomer attention before pooling.
        Requires ``weights_col`` to be set in the dataset.
    apply_weighting_to_graph:
        Where in the forward pass to apply monomer weight vectors.
    seed:
        Random seed for reproducibility.
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
        n_polymer_descriptors: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self._name = "BaseNetwork"
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.pooling = Pooling(pooling) if isinstance(pooling, str) else pooling
        self.n_convolutions = n_convolutions
        self.embedding_dim = embedding_dim
        self.readout_layers = readout_layers
        self.problem_type = (
            ProblemType(problem_type) if isinstance(problem_type, str) else problem_type
        )
        self.n_classes = n_classes
        self.dropout = dropout
        self.cross_att = cross_att
        self.apply_weighting_to_graph = (
            ApplyWeightingToGraph(apply_weighting_to_graph)
            if isinstance(apply_weighting_to_graph, str)
            else apply_weighting_to_graph
        )
        self.n_polymer_descriptors = n_polymer_descriptors
        self.seed = seed
        self.losses = None

        self._seed_everything(seed)

        self.pooling_fn = POOLING_FUNCTIONS.get(self.pooling, gmeanp)
        self.graph_embedding = (
            self.embedding_dim * 2
            if self.pooling == Pooling.GlobalMeanMaxPool
            else self.embedding_dim
        )

    # ------------------------------------------------------------------
    # Layer construction (called by subclasses after conv_layers are set)
    # ------------------------------------------------------------------

    def make_readout_layers(self) -> None:
        """
        Build the MLP readout network.

        Must be called at the end of each subclass ``__init__`` after
        ``conv_layers`` and ``norm_layers`` have been defined.
        """
        self.readout = nn.ModuleList()
        dim = self.graph_embedding + self.n_polymer_descriptors

        for _ in range(self.readout_layers - 1):
            reduced = dim // 2
            self.readout.append(nn.Sequential(nn.Linear(dim, reduced), nn.BatchNorm1d(reduced)))
            dim = reduced

        self.output_layer = nn.Linear(dim, self.n_classes)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch_index: Tensor | None = None,
        edge_attr: Tensor | None = None,
        monomer_weight: Tensor | None = None,
        monomer_id: Tensor | None = None,
        polymer_descriptors: Tensor | None = None,
    ) -> Tensor:
        """
        Full forward pass: message passing → pooling → readout.

        Parameters
        ----------
        x:
            Node feature matrix of shape ``(n_nodes, n_node_features)``.
        edge_index:
            Edge connectivity in COO format, shape ``(2, n_edges)``.
        batch_index:
            Batch vector mapping each node to its graph in the batch.
        edge_attr:
            Edge feature matrix of shape ``(n_edges, n_edge_features)``.
        monomer_weight:
            Per-node monomer weight fractions of shape ``(n_nodes, 1)``.
        polymer_descriptors:
            Graph-level polymer descriptor tensor of shape
            ``(batch_size, n_polymer_descriptors)``. Concatenated to the
            pooled graph embedding before the FFN readout.

        Returns
        -------
        Tensor
            Predictions of shape ``(batch_size, n_classes)``.
        """
        embedding = self.get_graph_embedding(
            x=x,
            edge_index=edge_index,
            batch_index=batch_index,
            edge_attr=edge_attr,
            monomer_weight=monomer_weight,
            monomer_id=monomer_id,
        )
        if polymer_descriptors is not None and self.n_polymer_descriptors > 0:
            embedding = torch.cat([embedding, polymer_descriptors], dim=1)
        preds = self.readout_function(embedding)

        if self.n_classes == 1:
            preds = preds.float()

        return preds

    def _pool_per_monomer(
        self, x: Tensor, batch_index: Tensor, monomer_weight: Tensor, monomer_id: Tensor
    ) -> Tensor:
        """
        "Ideal" per-monomer pooling.

        Pools each monomer's nodes *separately* using the configured
        pooling function, multiplies each monomer's pooled embedding by
        its molar weight, then sums the weighted per-monomer embeddings
        within each polymer:

            embedding(polymer) = Σ_m  w_m · pool(nodes of monomer m)

        This is invariant to monomer atom counts and to zero-weight
        monomers (a zero-weight monomer contributes ``0`` to the sum and
        nothing else), and matches the user's mental model of a copolymer
        as a weighted blend of monomer-level representations. Works for
        any pooling style (add / mean / max / mean-max) because the same
        ``pooling_fn`` is applied per monomer; the output dimensionality
        is unchanged.
        """
        # Unique consecutive segment id per (polymer, monomer).
        m_id = monomer_id.view(-1).long()
        n_mon = int(m_id.max().item()) + 1 if m_id.numel() > 0 else 1
        seg_raw = batch_index * n_mon + m_id
        _, seg = torch.unique(seg_raw, return_inverse=True)

        # Per-monomer embedding (configured pooling applied within each monomer).
        g_seg = self.pooling_fn(x, seg)
        # Per-monomer scalar weight (constant across a monomer's nodes;
        # mean of a constant returns that constant).
        w_seg = gmeanp(monomer_weight, seg)
        # Which polymer each segment belongs to (batch_index is constant
        # within a segment).
        poly_of_seg = gmeanp(batch_index.float().view(-1, 1), seg).round().long().view(-1)
        # Weighted sum of per-monomer embeddings, grouped back to polymers.
        return gap(g_seg * w_seg, poly_of_seg)

    def _pool(
        self,
        x: Tensor,
        batch_index: Tensor,
        monomer_weight: Tensor | None = None,
        monomer_id: Tensor | None = None,
    ) -> Tensor:
        """
        Pool node features into a graph-level embedding.

        ``PerMonomerPooling``: each monomer is pooled separately, weighted
        by its molar fraction, then summed (see ``_pool_per_monomer``).
        Falls back to plain pooling if the required per-node tensors are
        absent.

        ``BeforePooling``: when ``monomer_weight`` is provided (node
        features have just been multiplied by their monomer weight), the
        mean component of the pool is computed as a proper weighted mean —
        ``sum(weight * x) / sum(weight)`` per graph — matching the wD-MPNN
        readout convention from Aldeghi & Coley (2022,
        https://pubs.rsc.org/en/content/articlehtml/2022/sc/d2sc02839e,
        see ``polymer-chemprop/chemprop/models/mpn.py`` line 159). Sum and
        max pooling are unaffected — only the mean component (in
        ``GlobalMeanPool`` and ``GlobalMeanMaxPool``) is replaced.
        """
        if self.apply_weighting_to_graph == ApplyWeightingToGraph.PerMonomerPooling:
            if monomer_weight is None or monomer_id is None:
                return self.pooling_fn(x, batch_index)
            return self._pool_per_monomer(x, batch_index, monomer_weight, monomer_id)

        use_weighted_mean = (
            monomer_weight is not None
            and self.apply_weighting_to_graph == ApplyWeightingToGraph.BeforePooling
        )
        if not use_weighted_mean:
            return self.pooling_fn(x, batch_index)

        if self.pooling == Pooling.GlobalMeanPool:
            num = gap(x, batch_index)
            denom = gap(monomer_weight, batch_index).clamp(min=1e-8)
            return num / denom

        if self.pooling == Pooling.GlobalMeanMaxPool:
            num = gap(x, batch_index)
            denom = gap(monomer_weight, batch_index).clamp(min=1e-8)
            mean_part = num / denom
            max_part = gmp(x, batch_index)
            # Match the column order produced by ``_max_mean_pool``: [max, mean]
            return torch.cat([max_part, mean_part], dim=1)

        # GlobalAddPool / GlobalMaxPool: no normalization change needed
        return self.pooling_fn(x, batch_index)

    def get_graph_embedding(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch_index: Tensor | None = None,
        edge_attr: Tensor | None = None,
        monomer_weight: Tensor | None = None,
        monomer_id: Tensor | None = None,
    ) -> Tensor:
        """
        Run message passing and pooling to produce a graph-level embedding.

        Subclasses override this method when their convolution call signature
        differs (e.g. GCN does not pass ``edge_attr``).
        """
        if (
            monomer_weight is not None
            and self.apply_weighting_to_graph == ApplyWeightingToGraph.BeforeMPP
        ):
            x = x * monomer_weight

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

        return self._pool(x, batch_index, monomer_weight, monomer_id)

    def get_node_embeddings(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch_index: Tensor | None = None,
        edge_attr: Tensor | None = None,
        monomer_weight: Tensor | None = None,
        monomer_id: Tensor | None = None,
    ) -> Tensor:
        """
        Run message passing and return pre-pooling node embeddings.

        Identical to ``get_graph_embedding`` except the pooling step is
        omitted, returning the node-level tensor of shape
        ``(n_nodes, embedding_dim)`` that would otherwise be pooled.

        Subclasses that override ``get_graph_embedding`` must also override
        this method to apply the same pre-pooling transformations.
        """
        if (
            monomer_weight is not None
            and self.apply_weighting_to_graph == ApplyWeightingToGraph.BeforeMPP
        ):
            x = x * monomer_weight

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

        return x

    def readout_function(self, x: Tensor) -> Tensor:
        """Apply the MLP readout layers to a graph embedding."""
        for layer in self.readout:
            x = F.dropout(F.leaky_relu(layer(x)), p=self.dropout, training=self.training)
        return self.output_layer(x)

    # ------------------------------------------------------------------
    # Prediction interface
    # ------------------------------------------------------------------

    def predict(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch_index: Tensor | None = None,
        edge_attr: Tensor | None = None,
        monomer_weight: Tensor | None = None,
        monomer_id: Tensor | None = None,
        polymer_descriptors: Tensor | None = None,
    ) -> np.ndarray:
        """
        Run a forward pass and return numpy predictions.

        Parameters
        ----------
        x, edge_index, batch_index, edge_attr, monomer_weight, polymer_descriptors:
            Same as ``forward()``.

        Returns
        -------
        np.ndarray
            Flattened prediction array.
        """
        return (
            self.forward(
                x=x,
                edge_index=edge_index,
                batch_index=batch_index,
                edge_attr=edge_attr,
                monomer_weight=monomer_weight,
                monomer_id=monomer_id,
                polymer_descriptors=polymer_descriptors,
            )
            .detach()
            .numpy()
            .flatten()
        )

    def predict_loader(self, loader) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference over a DataLoader and return all predictions.

        Parameters
        ----------
        loader:
            PyG DataLoader yielding batched graph objects.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(sample_ids, predictions)`` — both flattened arrays.
        """
        device = torch.device("cpu")
        self.to(device)
        self.eval()

        y_pred, idx = [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = self.forward(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    batch_index=batch.batch,
                    edge_attr=batch.edge_attr,
                    monomer_weight=getattr(batch, "weight_monomer", None),
                    monomer_id=getattr(batch, "monomer_id", None),
                    polymer_descriptors=getattr(batch, "polymer_descriptors", None),
                )
                y_pred.append(out.cpu().detach().numpy().flatten())
                idx.append(batch.idx)

        return np.concatenate(idx), np.concatenate(y_pred)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    def reset_parameters(self) -> None:
        """Reinitialise all model parameters and reset the random seed."""
        self._seed_everything(self.seed)
        for module in self.children():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def _seed_everything(self, seed: int) -> None:
        seed_everything(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def _split_batched_atom_feats(self, node_feats: Tensor, batch_index: Tensor) -> list[Tensor]:
        """Split batched node features into per-graph lists."""
        return [node_feats[batch_index == i] for i in range(batch_index.max().item() + 1)]

    def _split_monomer_feats(self, node_feats: Tensor, weight_monomer: Tensor) -> list[Tensor]:
        """Split node features by unique monomer weight value."""
        unique_vals = torch.unique(weight_monomer)
        weight_monomer = weight_monomer.squeeze()
        return [node_feats[weight_monomer == v] for v in unique_vals]

    def _cross_attention(self, x: Tensor, batch_index: Tensor, monomer_weight: Tensor) -> Tensor:
        """Apply pairwise cross-monomer attention."""
        split_feats = self._split_batched_atom_feats(x, batch_index)
        split_weights = self._split_batched_atom_feats(monomer_weight, batch_index)
        updated: list[Tensor] = []

        for poly_feats, poly_weights in zip(split_feats, split_weights):
            monomer_feats = self._split_monomer_feats(poly_feats, poly_weights)
            q1 = F.leaky_relu(self.monomer_W_att(monomer_feats[0]))
            q2 = F.leaky_relu(self.monomer_W_att(monomer_feats[1]))
            attn = torch.sigmoid(torch.matmul(q1, q2.t()))
            new_m1 = monomer_feats[0] + torch.matmul(attn, q2)
            new_m2 = monomer_feats[1] + torch.matmul(attn.t(), q1)
            updated.append(torch.cat([new_m1, new_m2], dim=0))

        return torch.cat(updated, dim=0)


# ---------------------------------------------------------------------------
# Base classification network
# ---------------------------------------------------------------------------


class BaseNetworkClassifier(BaseNetwork):
    """
    Base GNN model for polymer property classification.

    Extends ``BaseNetwork`` with softmax prediction, class probability
    output, and a DataLoader inference method that returns both hard
    predictions and soft probability scores.
    """

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("problem_type", ProblemType.Classification)
        super().__init__(**kwargs)
        self._name = "BaseNetworkClassifier"

    def predict(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch_index: Tensor | None = None,
        edge_attr: Tensor | None = None,
        monomer_weight: Tensor | None = None,
        monomer_id: Tensor | None = None,
        polymer_descriptors: Tensor | None = None,
    ) -> np.ndarray:
        """Return hard class predictions (argmax of softmax)."""
        probs = self.predict_probs(
            x=x,
            edge_index=edge_index,
            batch_index=batch_index,
            edge_attr=edge_attr,
            monomer_weight=monomer_weight,
            monomer_id=monomer_id,
            polymer_descriptors=polymer_descriptors,
        )
        return np.argmax(probs, axis=1)

    def predict_probs(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch_index: Tensor | None = None,
        edge_attr: Tensor | None = None,
        monomer_weight: Tensor | None = None,
        monomer_id: Tensor | None = None,
        polymer_descriptors: Tensor | None = None,
    ) -> np.ndarray:
        """Return softmax class probabilities."""
        out = self.forward(
            x=x,
            edge_index=edge_index,
            batch_index=batch_index,
            edge_attr=edge_attr,
            monomer_weight=monomer_weight,
            monomer_id=monomer_id,
            polymer_descriptors=polymer_descriptors,
        )
        return torch.softmax(out, dim=1).detach().numpy()

    def predict_loader(self, loader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference over a DataLoader for classification.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(sample_ids, hard_predictions, class_probabilities)``
        """
        device = torch.device("cpu")
        self.to(device)
        self.eval()

        y_pred, idx, y_score = [], [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = (
                    self.forward(
                        x=batch.x,
                        edge_index=batch.edge_index,
                        batch_index=batch.batch,
                        edge_attr=batch.edge_attr,
                        monomer_weight=getattr(batch, "weight_monomer", None),
                        monomer_id=getattr(batch, "monomer_id", None),
                        polymer_descriptors=getattr(batch, "polymer_descriptors", None),
                    )
                    .cpu()
                    .detach()
                )

                probs = torch.softmax(out, dim=1)
                preds = torch.argmax(probs, dim=1)

                y_score.append(probs.numpy())
                y_pred.append(preds.numpy().flatten())
                idx.append(batch.idx)

        return (np.concatenate(idx), np.concatenate(y_pred), np.concatenate(y_score))
