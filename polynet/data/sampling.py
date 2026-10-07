"""
polynet.data.sampling
=====================
Inputs and provenance of the astartes sampler that draws the data splits.

Fingerprint samplers (``kennard_stone``, ``spxy``, ``kmeans``, ...) place each
polymer in chemical space by a *sampling fingerprint*: the ratio-weighted
average of its monomers' count fingerprints, computed with the settings of
``splitting.sampling_fingerprint``. This fingerprint is used for splitting
only; it never changes the model representations.

Public API
----------
::

    from polynet.data.sampling import sampling_features, sampling_metadata
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import logging

import numpy as np
import pandas as pd

from polynet.config.constants import POLYBERT_MODEL
from polynet.config.schemas.split_data import (
    DETERMINISTIC_SAMPLERS,
    SAMPLERS_USING_FINGERPRINTS,
    SAMPLERS_USING_TARGET,
    SplitConfig,
)

logger = logging.getLogger(__name__)


def sampling_features(
    data: pd.DataFrame,
    smiles_cols: list[str],
    weights_col: dict[str, str] | None,
    split_cfg: SplitConfig,
) -> np.ndarray | None:
    """
    Sampling fingerprints of the polymers in ``data``, if the sampler needs them.

    Parameters
    ----------
    data:
        Dataset with the SMILES (and ratio) columns, in the order the split
        is computed.
    smiles_cols:
        Monomer SMILES columns.
    weights_col:
        Mapping from SMILES column to its ratio column (``None``: equal
        weights).
    split_cfg:
        Splitting settings (``sampler`` and ``sampling_fingerprint``).

    Returns
    -------
    np.ndarray or None
        Array of shape ``(len(data), fp_size)``, or ``None`` when the sampler
        does not use fingerprints.
    """
    if split_cfg.sampler not in SAMPLERS_USING_FINGERPRINTS:
        return None
    # Deferred import: the featurizer pulls in RDKit / polyBERT machinery.
    from polynet.featurizer.descriptors import (
        polymer_count_fingerprints,
        polymer_polybert_fingerprints,
    )

    fp_cfg = split_cfg.sampling_fingerprint
    settings = fp_cfg.settings().model_dump() if fp_cfg.is_count_fingerprint else POLYBERT_MODEL
    logger.info(
        f"Computing {fp_cfg.fingerprint.value} sampling fingerprints ({settings}) for the "
        f"'{split_cfg.sampler.value}' sampler; they are used for splitting only."
    )
    if fp_cfg.is_count_fingerprint:
        return polymer_count_fingerprints(
            data, smiles_cols, weights_col, fp_cfg.fingerprint, fp_cfg.settings()
        )
    return polymer_polybert_fingerprints(data, smiles_cols, weights_col)


def sampling_metadata(split_cfg: SplitConfig, weights_col: dict[str, str] | None) -> dict:
    """
    Record of how the splits were sampled, saved next to the split indices.

    Parameters
    ----------
    split_cfg:
        Splitting settings.
    weights_col:
        Ratio columns used to weight the monomer fingerprints.

    Returns
    -------
    dict
        The sampler, its library and version, whether it uses fingerprints
        or targets and is deterministic, and the sampling fingerprint
        settings (``None`` when no fingerprint is used).
    """
    try:
        astartes_version = version("astartes")
    except PackageNotFoundError:
        astartes_version = None

    fp_cfg = split_cfg.sampling_fingerprint
    return {
        "library": "astartes",
        "astartes_version": astartes_version,
        "sampler": split_cfg.sampler.value,
        "sampler_hyperparameters": "astartes defaults",
        "split_method": split_cfg.split_method.value,
        "uses_fingerprints": split_cfg.sampler in SAMPLERS_USING_FINGERPRINTS,
        "uses_target": split_cfg.sampler in SAMPLERS_USING_TARGET,
        "deterministic": split_cfg.sampler in DETERMINISTIC_SAMPLERS,
        "sampling_fingerprint": (
            None
            if fp_cfg is None
            else {
                "fingerprint": fp_cfg.fingerprint.value,
                **(
                    fp_cfg.settings().model_dump()
                    if fp_cfg.is_count_fingerprint
                    else {"model": POLYBERT_MODEL}
                ),
                "merging": "ratio-weighted average" if weights_col else "equal-weight average",
                "weights_col": weights_col,
            }
        ),
    }
