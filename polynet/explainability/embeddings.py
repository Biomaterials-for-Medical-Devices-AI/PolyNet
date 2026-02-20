"""
polynet.explainability.embeddings
==================================
Graph embedding extraction and dimensionality reduction for GNN analysis.

Extracts latent graph representations from trained GNN models and reduces
them to 2D for visualisation using PCA or t-SNE. Both functions are pure
computation — all Streamlit rendering is handled by the calling app layer.

Public API
----------
::

    from polynet.explainability.embeddings import get_graph_embeddings, reduce_embeddings
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader

from polynet.config.enums import DimensionalityReduction

logger = logging.getLogger(__name__)


def get_graph_embeddings(dataset, model) -> pd.DataFrame:
    """
    Extract graph-level embeddings from a trained GNN model.

    Runs the model's message passing and pooling stages (``get_graph_embedding``)
    without the readout MLP, producing a latent vector per polymer graph.

    Parameters
    ----------
    dataset:
        PyG dataset or list of graph objects.
    model:
        Trained GNN model with a ``get_graph_embedding`` method.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape ``(n_graphs, embedding_dim)`` indexed by
        graph sample IDs.
    """
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    embeddings: list[np.ndarray] = []
    ids: list = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            embedding = model.get_graph_embedding(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch_index=batch.batch,
                monomer_weight=batch.weight_monomer,
            )
            embeddings.append(embedding.cpu().numpy())
            ids.append(batch.idx)

    flat_ids = np.array(ids).flatten().tolist()
    embedding_matrix = np.concatenate(embeddings, axis=0)

    return pd.DataFrame(embedding_matrix, index=flat_ids)


def reduce_embeddings(
    embeddings: pd.DataFrame,
    method: DimensionalityReduction | str,
    method_params: dict | None = None,
) -> pd.DataFrame:
    """
    Reduce high-dimensional graph embeddings to 2D for visualisation.

    Parameters
    ----------
    embeddings:
        DataFrame of shape ``(n_samples, embedding_dim)`` as returned by
        ``get_graph_embeddings``.
    method:
        Dimensionality reduction algorithm to apply.
    method_params:
        Optional dict of additional keyword arguments forwarded to the
        sklearn estimator (e.g. ``{"perplexity": 30}`` for t-SNE).

    Returns
    -------
    pd.DataFrame
        DataFrame of shape ``(n_samples, 2)`` with columns ``["Dim1", "Dim2"]``,
        preserving the original index.

    Raises
    ------
    ValueError
        If ``method`` is not a recognised ``DimensionalityReduction`` value.
    """
    method = DimensionalityReduction(method) if isinstance(method, str) else method
    params = method_params or {}

    if method == DimensionalityReduction.tSNE:
        reducer = TSNE(n_components=2, **params)
    elif method == DimensionalityReduction.PCA:
        reducer = PCA(n_components=2, **params)
    else:
        raise ValueError(
            f"Unknown dimensionality reduction method '{method}'. "
            f"Available: {[m.value for m in DimensionalityReduction]}."
        )

    reduced = reducer.fit_transform(embeddings.to_numpy())
    return pd.DataFrame(reduced, index=embeddings.index, columns=["Dim1", "Dim2"])
