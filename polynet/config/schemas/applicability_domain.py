"""
polynet.config.schemas.applicability_domain
===========================================
Pydantic schema for the applicability domain (AD) assessed when predicting
new data.

Each polymer is placed relative to the training polymers with one or both
distances of ``metric``: the min–max (Ruzicka) distance on a ratio-weighted
count fingerprint (same settings schema as the splitting
``sampling_fingerprint``, count fingerprints only), the same for every model;
or the Euclidean distance in each traditional model's own inputs. Its domain
is assessed with the k-nearest-neighbour approach of Tropsha, Gramatica &
Gombar (QSAR Comb. Sci. 2003, 22, 69–77), and the expected error at its
distance is read from the held-out test errors (Sheridan et al., J. Chem. Inf. Comput. Sci. 2004, 44, 1912–1928).
"""

from typing import Any

from pydantic import Field, field_validator, model_validator

from polynet.config.enums import ApplicabilityDomainMetric
from polynet.config.schemas.base import PolynetBaseModel
from polynet.config.schemas.fingerprints import SamplingFingerprintConfig


class ApplicabilityDomainConfig(PolynetBaseModel):
    """
    Applicability domain settings for predictions on new data.

    Attributes
    ----------
    enabled:
        Assess the applicability domain when predicting new data.
    metric:
        Distance(s) to assess the domain with (one or a list):
        ``ruzicka_morgan`` (default; Ruzicka distance on ``fingerprint``, one
        domain per model family) and/or ``euclidean_model_inputs`` (Euclidean
        distance in the scaled, feature-selected inputs of the traditional
        models, one domain per representation; GNNs only use
        ``ruzicka_morgan``). Each gives its own set of columns.
    k_neighbours:
        Number of nearest training polymers ``k`` whose mean distance places
        a polymer relative to the training set.
    z:
        Cutoff parameter ``Z``: a polymer is in domain when its distance is at
        most ``<d> + Z·σ``, with ``<d>`` and ``σ`` the mean and standard
        deviation of the training polymers' own k-nearest-neighbour distances.
    n_bins:
        Number of bins of domain score (quantiles of the test-set scores)
        used to estimate the expected error at a given distance.
    fingerprint:
        Count fingerprint of ``ruzicka_morgan`` (``morgan`` or ``rdkitfp``;
        defaults: Morgan, 2048 bins, radius 3).
    """

    enabled: bool = Field(default=True, description="Assess the applicability domain.")
    metric: list[ApplicabilityDomainMetric] = Field(
        default_factory=lambda: [ApplicabilityDomainMetric.RuzickaMorgan],
        min_length=1,
        description="Distance(s) used for the domain.",
    )
    k_neighbours: int = Field(default=5, ge=1, description="Nearest training polymers used.")
    z: float = Field(default=0.5, ge=0.0, description="Cutoff parameter Z of <d> + Z·σ.")
    n_bins: int = Field(default=5, ge=1, description="Score bins for the expected error.")
    fingerprint: SamplingFingerprintConfig = Field(
        default_factory=SamplingFingerprintConfig,
        description="Count fingerprint of the ruzicka_morgan distance.",
    )

    @field_validator("metric", mode="before")
    @classmethod
    def metric_as_list(cls, value: Any) -> Any:
        """Accept one metric or a list; drop repeats."""
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            return list(dict.fromkeys(value))
        return value

    @model_validator(mode="after")
    def count_fingerprint_only(self) -> "ApplicabilityDomainConfig":
        """Ruzicka similarity needs non-negative counts, which polyBERT does not give."""
        if not self.fingerprint.is_count_fingerprint:
            raise ValueError(
                "applicability_domain.fingerprint must be a count fingerprint (morgan or "
                f"rdkitfp), got '{self.fingerprint.fingerprint.value}'."
            )
        return self
