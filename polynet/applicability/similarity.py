"""
polynet.applicability.similarity
================================
Distances between polymers and k-nearest-neighbour distance to a reference set.

- **Ruzicka** (min–max) similarity of count fingerprints,
  ``Σ min(a, b) / Σ max(a, b)``: the generalisation of the Tanimoto
  coefficient to non-negative counts (RDKit's count Tanimoto). It is 1 for
  identical polymers and 0 for polymers that share no fingerprint bin; the
  distance is 1 − similarity.
- **Euclidean** distance, for real-valued model inputs (e.g. scaled
  descriptors).

::

    from polynet.applicability.similarity import knn_distance, ruzicka_similarity
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from polynet.config.enums import ApplicabilityDomainMetric

# Size of the (rows × reference × bins) block compared at once, in bytes.
_BLOCK_BYTES = 64 * 1024**2


def ruzicka_similarity(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Ruzicka (min–max) similarity between every query and every reference fingerprint.

    Parameters
    ----------
    query:
        Array of shape ``(n_query, n_bins)`` with non-negative counts.
    reference:
        Array of shape ``(n_reference, n_bins)`` with non-negative counts.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_query, n_reference)`` with values in [0, 1]. Pairs
        of empty fingerprints have similarity 0.

    Raises
    ------
    ValueError
        If the fingerprints have different lengths or negative values.
    """
    query = np.asarray(query, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError(
            f"Fingerprints must be 2D with the same number of bins, got {query.shape} "
            f"and {reference.shape}."
        )
    if (query < 0).any() or (reference < 0).any():
        raise ValueError("Ruzicka similarity needs non-negative fingerprint counts.")

    # Bins empty in both sets add nothing to Σmin or Σmax.
    used = (query.sum(axis=0) > 0) | (reference.sum(axis=0) > 0)
    query, reference = query[:, used], reference[:, used]

    totals = query.sum(axis=1)[:, None] + reference.sum(axis=1)[None, :]
    shared = np.empty((len(query), len(reference)))
    rows = max(1, _BLOCK_BYTES // max(1, reference.size * 8))
    for start in range(0, len(query), rows):
        block = query[start : start + rows]
        shared[start : start + rows] = np.minimum(block[:, None, :], reference[None]).sum(axis=2)

    union = totals - shared  # Σmax = Σa + Σb − Σmin
    return np.divide(shared, union, out=np.zeros_like(shared), where=union > 0)


def knn_similarity(
    query: np.ndarray, reference: np.ndarray, k: int, exclude_self: bool = False
) -> np.ndarray:
    """
    Mean Ruzicka similarity of each query to its ``k`` most similar references.

    Equals ``1 − knn_distance(..., RuzickaMorgan)``; see ``knn_distance``.
    """
    return 1.0 - knn_distance(
        query, reference, k, ApplicabilityDomainMetric.RuzickaMorgan, exclude_self
    )


def knn_distance(
    query: np.ndarray,
    reference: np.ndarray,
    k: int,
    metric: ApplicabilityDomainMetric | str = ApplicabilityDomainMetric.RuzickaMorgan,
    exclude_self: bool = False,
) -> np.ndarray:
    """
    Mean distance of each query to its ``k`` nearest references.

    Parameters
    ----------
    query, reference:
        Arrays of shape ``(n, n_features)``: count fingerprints for
        ``ruzicka_morgan``, any real values for ``euclidean_model_inputs``.
    k:
        Number of nearest neighbours; capped at the number of available
        references.
    metric:
        ``ruzicka_morgan`` (1 − Ruzicka similarity) or
        ``euclidean_model_inputs`` (Euclidean distance).
    exclude_self:
        ``query`` *is* ``reference`` (same rows, same order) and each row must
        not be its own neighbour — used to place the reference set relative
        to itself.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_query,)``.

    Raises
    ------
    ValueError
        If ``k < 1`` or there are no references to compare with.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}.")
    if ApplicabilityDomainMetric(metric) == ApplicabilityDomainMetric.RuzickaMorgan:
        distance = 1.0 - ruzicka_similarity(query, reference)
    else:
        distance = cdist(np.asarray(query, dtype=float), np.asarray(reference, dtype=float))
    if exclude_self:
        if distance.shape[0] != distance.shape[1]:
            raise ValueError("exclude_self needs query and reference to be the same rows.")
        np.fill_diagonal(distance, np.inf)
    available = distance.shape[1] - int(exclude_self)
    if available < 1:
        raise ValueError("There are no reference polymers to compare with.")
    k = min(k, available)
    return np.partition(distance, k - 1, axis=1)[:, :k].mean(axis=1)
