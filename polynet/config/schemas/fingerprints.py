"""
polynet.config.schemas.fingerprints
===================================
Pydantic schemas for the settings of count fingerprints (Morgan, RDKit).

In the config, a fingerprint is requested under
``representations.molecular_descriptors`` either with ``true`` (or ``[]``) for
the defaults, or with a mapping of settings::

    molecular_descriptors:
      morgan: {fp_size: 1024, radius: 2}
      rdkitfp: true

The defaults are RDKit's own ``rdFingerprintGenerator`` defaults, which
PolyNet has always used: 2048 bins, and Morgan radius 3 (ECFP6-like).
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from polynet.config.enums import MolecularDescriptor
from polynet.config.schemas.base import PolynetBaseModel


class CountFingerprintConfig(PolynetBaseModel):
    """Settings shared by all count fingerprints."""

    fp_size: int = Field(default=2048, ge=1, description="Fingerprint length (number of bins).")

    @classmethod
    def from_value(cls, value: Any) -> "CountFingerprintConfig":
        """
        Build the settings from a ``molecular_descriptors`` value.

        ``True``, ``None``, an empty list or an empty dict select the defaults
        (the forms used by older configs and saved experiments); a mapping
        overrides individual settings.

        Raises
        ------
        ValueError
            If ``value`` is neither of those.
        """
        if value is None or value is True or value == [] or value == {}:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise ValueError(
            f"Fingerprint settings must be true or a mapping of "
            f"{sorted(cls.model_fields)}, got {value!r}."
        )


class MorganFingerprintConfig(CountFingerprintConfig):
    """Morgan count fingerprint settings."""

    radius: int = Field(
        default=3, ge=0, description="Radius of the atom environments (3 ≈ ECFP6, 2 ≈ ECFP4)."
    )


class RDKitFingerprintConfig(CountFingerprintConfig):
    """RDKit (path-based) count fingerprint settings."""


FINGERPRINT_CONFIGS: dict[MolecularDescriptor, type[CountFingerprintConfig]] = {
    MolecularDescriptor.Morgan: MorganFingerprintConfig,
    MolecularDescriptor.RDKitFP: RDKitFingerprintConfig,
}


def resolve_fingerprint_config(
    descriptor: MolecularDescriptor | str, value: Any
) -> CountFingerprintConfig:
    """
    Return the validated settings of a count fingerprint.

    Parameters
    ----------
    descriptor:
        ``morgan`` or ``rdkitfp``.
    value:
        Its ``molecular_descriptors`` value (``true``, ``[]``, a mapping, …).

    Returns
    -------
    CountFingerprintConfig
        The settings, defaults filled in.

    Raises
    ------
    ValueError
        If the value or any setting is invalid (unknown setting, wrong type,
        ``fp_size < 1``, ``radius < 0``).
    """
    return FINGERPRINT_CONFIGS[MolecularDescriptor(descriptor)].from_value(value)


# Sampling fingerprints: the count fingerprints plus the PolyBERT embedding.
SAMPLING_FINGERPRINTS = frozenset({*FINGERPRINT_CONFIGS, MolecularDescriptor.PolyBERT})


class SamplingFingerprintConfig(PolynetBaseModel):
    """
    Fingerprint used only to place polymers in chemical space for data splitting.

    Each monomer's fingerprint is computed and the polymer fingerprint is their
    ratio-weighted average (as in the representations). These settings never
    change the model representations.

    ``morgan`` and ``rdkitfp`` are count fingerprints with ``fp_size`` (and,
    for Morgan, ``radius``); ``polybert`` is the fixed-size polyBERT embedding
    and takes no settings.
    """

    fingerprint: MolecularDescriptor = Field(
        default=MolecularDescriptor.Morgan, description="morgan, rdkitfp or polybert."
    )
    fp_size: int | None = Field(
        default=None, ge=1, description="Fingerprint length (default 2048); count fingerprints only."
    )
    radius: int | None = Field(
        default=None, ge=0, description="Morgan radius (default 3); morgan only."
    )

    @model_validator(mode="after")
    def check_fingerprint(self) -> "SamplingFingerprintConfig":
        if self.fingerprint not in SAMPLING_FINGERPRINTS:
            raise ValueError(
                f"sampling_fingerprint.fingerprint must be one of "
                f"{sorted(str(f) for f in SAMPLING_FINGERPRINTS)}, got '{self.fingerprint}'."
            )
        if self.fingerprint == MolecularDescriptor.PolyBERT:
            if self.fp_size is not None or self.radius is not None:
                raise ValueError(
                    "sampling_fingerprint: polybert is a fixed-size embedding; fp_size and "
                    "radius do not apply."
                )
            return self
        if self.fp_size is None:
            self.fp_size = CountFingerprintConfig().fp_size
        if self.fingerprint == MolecularDescriptor.Morgan and self.radius is None:
            self.radius = MorganFingerprintConfig().radius
        if self.fingerprint != MolecularDescriptor.Morgan and self.radius is not None:
            raise ValueError("sampling_fingerprint.radius only applies to morgan fingerprints.")
        return self

    @property
    def is_count_fingerprint(self) -> bool:
        """Whether this is a count fingerprint (``morgan`` / ``rdkitfp``)."""
        return self.fingerprint in FINGERPRINT_CONFIGS

    def settings(self) -> CountFingerprintConfig:
        """
        Count fingerprint settings (``MorganFingerprintConfig`` / ``RDKitFingerprintConfig``).

        Raises
        ------
        ValueError
            For ``polybert``, which has no settings.
        """
        if not self.is_count_fingerprint:
            raise ValueError(f"'{self.fingerprint.value}' has no fingerprint settings.")
        values = {"fp_size": self.fp_size}
        if self.radius is not None:
            values["radius"] = self.radius
        return FINGERPRINT_CONFIGS[self.fingerprint](**values)
