"""
polynet.applicability
=====================
Applicability domain of trained models for new polymers: whether a new
polymer is similar enough to the training polymers for the models to be
trusted, and the error to expect at its distance from them.

::

    from polynet.applicability import ModelGroup, assess_applicability_domain
"""

from polynet.applicability.assess import ModelGroup, assess_applicability_domain, summarise_domain
from polynet.applicability.calibration import DistanceErrorCalibration
from polynet.applicability.domain import KNNApplicabilityDomain
from polynet.applicability.similarity import knn_distance, knn_similarity, ruzicka_similarity

__all__ = [
    "KNNApplicabilityDomain",
    "ModelGroup",
    "DistanceErrorCalibration",
    "assess_applicability_domain",
    "knn_distance",
    "knn_similarity",
    "ruzicka_similarity",
    "summarise_domain",
]
