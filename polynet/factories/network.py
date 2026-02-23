"""
polynet.factories.network
==========================
Factory functions for constructing GNN model instances.

Design
------
Models are looked up via a dispatch table keyed by ``(Network, ProblemType)``
rather than a long if/elif chain. Adding a new architecture requires only:

1. Implementing the model class in ``polynet.models``
2. Adding two entries to ``_NETWORK_REGISTRY`` (one per problem type)

Public API
----------
The primary entry point is ``create_network``, which accepts individual
arguments so that library users can call it directly without constructing
a full config object::

    from polynet.factories.network import create_network
    from polynet.config.enums import Network, ProblemType

    model = create_network(
        network=Network.GCN,
        problem_type=ProblemType.Regression,
        num_node_features=10,
        num_edge_features=5,
        embedding_dim=64,
        n_convolutions=3,
        readout_layers=2,
        dropout=0.05,
    )
"""

from __future__ import annotations

from typing import Any

from polynet.config.enums import Network, ProblemType
from polynet.models.CGGNN import CGGNNClassifier, CGGNNRegressor
from polynet.models.GAT import GATClassifier, GATRegressor
from polynet.models.GCN import GCNClassifier, GCNRegressor
from polynet.models.MPNN import MPNNClassifier, MPNNRegressor
from polynet.models.TransfomerGNN import TransformerGNNClassifier, TransformerGNNRegressor
from polynet.models.graphsage import GraphSageClassifier, GraphSageRegressor


# ---------------------------------------------------------------------------
# Registry
# (Network, ProblemType) → model class
# To add a new architecture: add two entries here, one per problem type.
# ---------------------------------------------------------------------------

_NETWORK_REGISTRY: dict[tuple[Network, ProblemType], type] = {
    (Network.GCN, ProblemType.Classification): GCNClassifier,
    (Network.GCN, ProblemType.Regression): GCNRegressor,
    (Network.GraphSAGE, ProblemType.Classification): GraphSageClassifier,
    (Network.GraphSAGE, ProblemType.Regression): GraphSageRegressor,
    (Network.GAT, ProblemType.Classification): GATClassifier,
    (Network.GAT, ProblemType.Regression): GATRegressor,
    (Network.TransformerGNN, ProblemType.Classification): TransformerGNNClassifier,
    (Network.TransformerGNN, ProblemType.Regression): TransformerGNNRegressor,
    (Network.MPNN, ProblemType.Classification): MPNNClassifier,
    (Network.MPNN, ProblemType.Regression): MPNNRegressor,
    (Network.CGGNN, ProblemType.Classification): CGGNNClassifier,
    (Network.CGGNN, ProblemType.Regression): CGGNNRegressor,
}


def create_network(network: Network | str, problem_type: ProblemType | str, **kwargs: Any) -> Any:
    """
    Construct and return a GNN model instance.

    Parameters
    ----------
    network:
        The GNN architecture to instantiate. Accepts a ``Network`` enum
        member or its string value (e.g. ``"GCN"``).
    problem_type:
        The supervised task type. Accepts a ``ProblemType`` enum member
        or its string value (e.g. ``"regression"``).
    **kwargs:
        All remaining keyword arguments are forwarded directly to the
        model constructor. See the individual model classes in
        ``polynet.models`` for the full list of accepted parameters.

    Returns
    -------
    nn.Module
        An instantiated, untrained GNN model.

    Raises
    ------
    ValueError
        If the ``(network, problem_type)`` combination is not registered.

    Examples
    --------
    >>> from polynet.factories.network import create_network
    >>> from polynet.config.enums import Network, ProblemType
    >>> model = create_network(
    ...     network=Network.GCN,
    ...     problem_type=ProblemType.Regression,
    ...     num_node_features=10,
    ...     embedding_dim=64,
    ...     n_convolutions=3,
    ...     readout_layers=2,
    ...     dropout=0.05,
    ... )
    """
    # Coerce strings to enums so callers can use either form
    network = Network(network) if isinstance(network, str) else network
    problem_type = ProblemType(problem_type) if isinstance(problem_type, str) else problem_type

    key = (network, problem_type)

    if key not in _NETWORK_REGISTRY:
        available = [f"({n.value}, {p.value})" for n, p in _NETWORK_REGISTRY if n == network]
        if available:
            raise ValueError(
                f"Network '{network.value}' does not support problem type "
                f"'{problem_type.value}'. "
                f"Supported problem types for this network: {available}."
            )
        raise ValueError(
            f"Network '{network.value}' is not registered. "
            f"Available networks: {sorted({n.value for n, _ in _NETWORK_REGISTRY})}."
        )

    model_cls = _NETWORK_REGISTRY[key]
    return model_cls(**kwargs)


def list_available_networks() -> dict[str, list[str]]:
    """
    Return a mapping of network name → supported problem types.

    Useful for introspection and documentation generation.

    Returns
    -------
    dict[str, list[str]]
        E.g. ``{"GCN": ["classification", "regression"], ...}``
    """
    result: dict[str, list[str]] = {}
    for network, problem_type in _NETWORK_REGISTRY:
        result.setdefault(network.value, []).append(problem_type.value)
    return {k: sorted(v) for k, v in sorted(result.items())}
