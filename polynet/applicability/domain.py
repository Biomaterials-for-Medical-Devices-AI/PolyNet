"""
polynet.applicability.domain
============================
k-nearest-neighbour applicability domain (Tropsha, Gramatica & Gombar,
QSAR Comb. Sci. 2003, 22, 69–77).

The reference (training) polymers are first placed relative to each other:
for each one, ``d`` is its mean distance to its ``k`` nearest other reference
polymers. A new polymer is in the domain when its own mean distance to its
``k`` nearest reference polymers is at most ``<d> + Z·σ``, with ``<d>`` and
``σ`` the mean and standard deviation of the reference distances. Its
**score** is that distance divided by the cutoff, so 1 is the domain boundary
whatever the distance (Ruzicka or Euclidean). When numeric polymer
descriptors (e.g. Mw) are also model inputs, the polymer must additionally lie
within their training range.

::

    from polynet.applicability.domain import KNNApplicabilityDomain
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from polynet.applicability.similarity import knn_distance
from polynet.config.enums import ApplicabilityDomainMetric


class KNNApplicabilityDomain:
    """
    k-nearest-neighbour applicability domain.

    Parameters
    ----------
    k:
        Number of nearest reference polymers.
    z:
        Cutoff parameter ``Z`` of ``<d> + Z·σ``.
    metric:
        ``ruzicka_morgan`` (count fingerprints) or ``euclidean_model_inputs``
        (real-valued model inputs).

    Attributes
    ----------
    reference_distance_mean_, reference_distance_std_:
        ``<d>`` and ``σ`` of the reference polymers' k-nearest-neighbour
        distances.
    cutoff_distance_:
        ``<d> + Z·σ``.
    descriptor_min_, descriptor_max_:
        Training range of the polymer descriptors (``None`` without
        descriptors).
    """

    def __init__(
        self,
        k: int = 5,
        z: float = 0.5,
        metric: ApplicabilityDomainMetric | str = ApplicabilityDomainMetric.RuzickaMorgan,
    ) -> None:
        self.k = k
        self.z = z
        self.metric = ApplicabilityDomainMetric(metric)

    def fit(
        self, features: np.ndarray, descriptors: pd.DataFrame | None = None
    ) -> "KNNApplicabilityDomain":
        """
        Learn the domain of the reference polymers.

        Parameters
        ----------
        features:
            Reference polymers, shape ``(n_reference, n_features)``.
        descriptors:
            Numeric polymer descriptors of the same polymers (optional).

        Raises
        ------
        ValueError
            With fewer than two reference polymers.
        """
        features = np.asarray(features, dtype=float)
        if len(features) < 2:
            raise ValueError("The applicability domain needs at least two reference polymers.")
        self.reference_ = features
        distances = knn_distance(features, features, self.k, self.metric, exclude_self=True)
        self.reference_distance_mean_ = float(distances.mean())
        self.reference_distance_std_ = float(distances.std())
        self.cutoff_distance_ = (
            self.reference_distance_mean_ + self.z * self.reference_distance_std_
        )
        if descriptors is not None and descriptors.shape[1]:
            self.descriptor_min_ = descriptors.min()
            self.descriptor_max_ = descriptors.max()
        else:
            self.descriptor_min_ = self.descriptor_max_ = None
        return self

    def score(self, features: np.ndarray, descriptors: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Place new polymers relative to the reference polymers.

        Parameters
        ----------
        features:
            New polymers, in the space the domain was fitted in.
        descriptors:
            Their polymer descriptors; required when the domain was fitted
            with descriptors.

        Returns
        -------
        pd.DataFrame
            One row per polymer (positional index) with ``distance`` (mean
            distance to the ``k`` nearest reference polymers), ``score``
            (distance / cutoff; above 1 is outside the domain),
            ``descriptors_in_range`` (``True`` without descriptors) and
            ``in_domain``.
        """
        distance = knn_distance(features, self.reference_, self.k, self.metric)
        in_range = np.ones(len(distance), dtype=bool)
        if self.descriptor_min_ is not None:
            if descriptors is None:
                raise ValueError("The domain was fitted with polymer descriptors; pass them.")
            values = descriptors[self.descriptor_min_.index]
            in_range = (
                ((values >= self.descriptor_min_) & (values <= self.descriptor_max_))
                .all(axis=1)
                .to_numpy()
            )
        score = (
            distance / self.cutoff_distance_
            if self.cutoff_distance_ > 0
            else np.where(distance > 0, np.inf, 0.0)
        )
        return pd.DataFrame(
            {
                "distance": distance,
                "score": score,
                "descriptors_in_range": in_range,
                "in_domain": (score <= 1.0) & in_range,
            }
        )

    def summary(self) -> dict:
        """Fitted domain, for the saved report."""
        return {
            "metric": self.metric.value,
            "n_reference": len(self.reference_),
            "k": self.k,
            "z": self.z,
            "reference_distance_mean": self.reference_distance_mean_,
            "reference_distance_std": self.reference_distance_std_,
            "cutoff_distance": self.cutoff_distance_,
            "descriptor_range": (
                None
                if self.descriptor_min_ is None
                else {
                    col: [float(self.descriptor_min_[col]), float(self.descriptor_max_[col])]
                    for col in self.descriptor_min_.index
                }
            ),
        }
