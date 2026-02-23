"""
polynet.featurizer
==================
Molecular featurization utilities for the polynet pipeline.

::

    from polynet.featurizer import build_vector_representation, calculate_descriptors
    from polynet.featurizer import uncorrelated_features, diversity_filter
"""

from polynet.featurizer.descriptors import (
    build_vector_representation,
    calculate_descriptors,
    get_polybert_fingerprints,
)
from polynet.featurizer.graph import PolymerGraphDataset
from polynet.featurizer.polymer_graph import CustomPolymerGraph
from polynet.featurizer.selection import (
    diversity_filter,
    sequential_forward_selection,
    uncorrelated_features,
)

__all__ = [
    # Descriptor computation and merging
    "build_vector_representation",
    "calculate_descriptors",
    "get_polybert_fingerprints",
    # Graph datasets
    "PolymerGraphDataset",
    "CustomPolymerGraph",
    # Feature selection
    "uncorrelated_features",
    "diversity_filter",
    "sequential_forward_selection",
]
