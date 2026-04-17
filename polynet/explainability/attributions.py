"""
polynet.explainability.attributions
=====================================
Node attribution utilities for GNN explainability.

Provides helpers for merging ensemble attribution masks, computing per-feature
vector sizes, slicing masks to a single feature, and deep-merging attribution
cache dicts.

Public API
----------
::

    from polynet.explainability.attributions import (
        merge_attribution_masks,
        get_node_feat_vector_sizes,
        slice_masks_to_feature,
        deep_update,
    )
"""

from __future__ import annotations

from collections import defaultdict
import logging
from pathlib import Path  # noqa: F401 — kept for downstream imports

import numpy as np

from polynet.config.enums import ExplainAlgorithm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mask merging
# ---------------------------------------------------------------------------


def merge_attribution_masks(
    node_masks: dict, model_log_names: list[str]
) -> dict[str, dict[str, list]]:
    """
    Average attribution masks across multiple models for each molecule.

    Each model's masks are normalised by the global maximum absolute
    attribution across all molecules for that model+algorithm before
    averaging, so that models with different attribution scales contribute
    equally.

    Parameters
    ----------
    node_masks:
        Structure: ``{model_name: {model_number: {mol_id: {algorithm: mask}}}}``.
    model_log_names:
        List of model log names (``"{model_name}_{number}"``).

    Returns
    -------
    dict[str, dict[str, list]]
        Merged masks: ``{mol_id: {algorithm: averaged_mask_list}}``.
    """
    # Step 1: compute global max absolute value per (model_log_name, algorithm)
    global_max: dict[tuple, float] = defaultdict(float)

    for log_name in model_log_names:
        model_name, number = log_name.split("_", 1)
        model_dict = node_masks.get(model_name, {}).get(number, {})

        for mol_id, algos in model_dict.items():
            for algo_name, masks in algos.items():
                arr = np.array(masks, dtype=float)
                if arr.size == 0:
                    continue
                key = (log_name, algo_name)
                global_max[key] = max(global_max[key], float(np.max(np.abs(arr))))

    # Step 2: accumulate normalised masks
    accumulator: dict = defaultdict(lambda: defaultdict(lambda: None))
    counts: dict = defaultdict(lambda: defaultdict(int))

    for log_name in model_log_names:
        model_name, number = log_name.split("_", 1)
        model_dict = node_masks.get(model_name, {}).get(number, {})

        for mol_id, algos in model_dict.items():
            for algo_name, masks in algos.items():
                arr = np.array(masks, dtype=float)
                max_abs = global_max.get((log_name, algo_name), 0.0)

                if max_abs > 0:
                    arr = arr / max_abs

                if accumulator[mol_id][algo_name] is None:
                    accumulator[mol_id][algo_name] = arr.copy()
                else:
                    accumulator[mol_id][algo_name] += arr

                counts[mol_id][algo_name] += 1

    # Step 3: compute mean and convert to lists
    return {
        mol_id: {
            algo_name: (summed / counts[mol_id][algo_name]).tolist()
            for algo_name, summed in algos.items()
            if counts[mol_id][algo_name] > 0
        }
        for mol_id, algos in accumulator.items()
    }


# ---------------------------------------------------------------------------
# Feature slicing
# ---------------------------------------------------------------------------


def get_node_feat_vector_sizes(node_features: dict) -> dict[str, int]:
    """
    Compute the one-hot encoding length for each configured node feature.

    Parameters
    ----------
    node_features:
        Node feature configuration dict as stored in the config schema.
        Each entry maps a feature name to its allowable values and wildcard
        flag.

    Returns
    -------
    dict[str, int]
        Mapping from feature name to the number of dimensions it occupies
        in the node feature vector.
    """
    from polynet.config.constants import FeatureKey

    sizes: dict[str, int] = {}

    for feat_name, feat_config in node_features.items():
        if not feat_config:
            sizes[feat_name] = 1
        else:
            n_vals = len(feat_config[FeatureKey.AllowableVals])
            wildcard = int(bool(feat_config[FeatureKey.Wildcard]))
            sizes[feat_name] = n_vals + wildcard

    return sizes


def slice_masks_to_feature(
    node_masks: dict,
    mols: list,
    algorithm: ExplainAlgorithm | str,
    node_features: dict,
    feature_name: str,
) -> dict:
    """
    Restrict attribution masks to the dimensions of a single node feature.

    Parameters
    ----------
    node_masks:
        Merged masks dict as returned by ``merge_attribution_masks``.
    mols:
        List of graph objects being explained.
    algorithm:
        Attribution algorithm whose masks to slice.
    node_features:
        Node feature configuration dict.
    feature_name:
        Feature to isolate. Pass ``"All Features"`` to return masks unchanged.

    Returns
    -------
    dict
        Updated masks dict with feature-sliced arrays.
    """
    algorithm = ExplainAlgorithm(algorithm) if isinstance(algorithm, str) else algorithm

    if feature_name == "All Features":
        return node_masks

    feature_sizes = get_node_feat_vector_sizes(node_features)
    start = 0
    end = 0

    for feat, size in feature_sizes.items():
        end += size
        if feat == feature_name:
            break
        start = end

    result = dict(node_masks)
    for mol in mols:
        mask = np.array(node_masks[mol.idx][algorithm.value])
        result[mol.idx] = dict(node_masks[mol.idx])
        result[mol.idx][algorithm.value] = mask[:, start:end]

    return result


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------


def deep_update(original: dict, updates: dict) -> dict:
    """
    Recursively merge ``updates`` into ``original`` in-place.

    Parameters
    ----------
    original:
        Base dictionary to update.
    updates:
        Dictionary whose values override or extend ``original``.

    Returns
    -------
    dict
        The updated ``original`` dict (modified in-place).
    """
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(original.get(key), dict):
            deep_update(original[key], value)
        else:
            original[key] = value
    return original
