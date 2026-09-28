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

from pydantic import Field

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
