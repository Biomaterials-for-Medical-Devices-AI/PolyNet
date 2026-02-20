"""
polynet.explainability.attributions
=====================================
Node attribution calculation and ensemble mask merging for GNN explainability.

Supports multiple attribution algorithms via PyTorch Geometric's ``Explainer``
interface, with Captum-based methods (Saliency, IntegratedGradients, etc.)
and GNNExplainer.

Results are cached to disk as JSON so that expensive attribution computations
are not repeated across app sessions or pipeline re-runs.

Public API
----------
::

    from polynet.explainability.attributions import (
        build_explainer,
        calculate_attributions,
        merge_attribution_masks,
        get_node_feat_vector_sizes,
        slice_masks_to_feature,
    )
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import captum.attr
import numpy as np
from torch_geometric.explain import CaptumExplainer, Explainer, GNNExplainer, ModelConfig

from polynet.config.enums import ExplainAlgorithm, ProblemType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Explainer construction
# ---------------------------------------------------------------------------

# Maps ExplainAlgorithm → Captum attribution class (or None for GNNExplainer)
_CAPTUM_REGISTRY: dict[ExplainAlgorithm, type | None] = {
    ExplainAlgorithm.GNNExplainer: None,
    ExplainAlgorithm.ShapleyValueSampling: captum.attr.ShapleyValueSampling,
    ExplainAlgorithm.InputXGradient: captum.attr.InputXGradient,
    ExplainAlgorithm.Saliency: captum.attr.Saliency,
    ExplainAlgorithm.IntegratedGradients: captum.attr.IntegratedGradients,
    ExplainAlgorithm.Deconvolution: captum.attr.Deconvolution,
    ExplainAlgorithm.GuidedBackprop: captum.attr.GuidedBackprop,
}


def build_explainer(
    model, algorithm: ExplainAlgorithm | str, problem_type: ProblemType | str
) -> Explainer:
    """
    Build a PyG ``Explainer`` for the given model and algorithm.

    Parameters
    ----------
    model:
        Trained GNN model.
    algorithm:
        Attribution algorithm to use.
    problem_type:
        Classification or regression — sets the ModelConfig task.

    Returns
    -------
    Explainer
        A configured PyG Explainer ready to call on graph inputs.

    Raises
    ------
    ValueError
        If the algorithm is not registered.
    """
    algorithm = ExplainAlgorithm(algorithm) if isinstance(algorithm, str) else algorithm
    problem_type = ProblemType(problem_type) if isinstance(problem_type, str) else problem_type

    if algorithm not in _CAPTUM_REGISTRY:
        raise ValueError(
            f"Explain algorithm '{algorithm}' is not registered. "
            f"Available: {[a.value for a in _CAPTUM_REGISTRY]}."
        )

    task = (
        "multiclass_classification" if problem_type == ProblemType.Classification else "regression"
    )
    model_config = ModelConfig(mode=task, task_level="graph", return_type="raw")

    captum_cls = _CAPTUM_REGISTRY[algorithm]
    if captum_cls is None:
        pyg_algorithm = GNNExplainer(epochs=100)
    else:
        pyg_algorithm = CaptumExplainer(attribution_method=captum_cls)

    return Explainer(
        model=model,
        algorithm=pyg_algorithm,
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type=None,
        model_config=model_config,
    )


def build_explainers(
    models: dict, algorithm: ExplainAlgorithm | str, problem_type: ProblemType | str
) -> dict[str, Explainer]:
    """
    Build one ``Explainer`` per trained model.

    Parameters
    ----------
    models:
        Dict of ``{model_log_name: model}`` as returned by
        ``train_gnn_ensemble``.
    algorithm:
        Attribution algorithm to use.
    problem_type:
        Classification or regression.

    Returns
    -------
    dict[str, Explainer]
        Mapping from model log name to its Explainer.
    """
    return {name: build_explainer(model, algorithm, problem_type) for name, model in models.items()}


# ---------------------------------------------------------------------------
# Attribution calculation
# ---------------------------------------------------------------------------


def calculate_attributions(
    mols: list,
    existing_explanations: dict,
    algorithm: ExplainAlgorithm | str,
    explainers: dict[str, Explainer],
) -> dict:
    """
    Calculate node attribution masks for a set of molecules.

    Skips molecules whose attributions are already cached in
    ``existing_explanations`` to avoid redundant computation.

    Parameters
    ----------
    mols:
        List of PyG graph objects to explain.
    existing_explanations:
        Previously computed explanations loaded from the cache JSON.
        Structure: ``{model_name: {model_number: {mol_id: {algorithm: mask}}}}``.
    algorithm:
        Attribution algorithm being computed.
    explainers:
        Dict of ``{"{model_name}_{number}": Explainer}`` as returned by
        ``build_explainers``.

    Returns
    -------
    dict
        Node masks with structure:
        ``{model_name: {model_number: {mol_id: {algorithm: mask}}}}``.
        Mask values are nested lists of shape ``(n_atoms, n_node_features)``.
    """
    algorithm = ExplainAlgorithm(algorithm) if isinstance(algorithm, str) else algorithm
    node_masks: dict = {}

    for model_log_name, explainer in explainers.items():
        model_name, model_number = model_log_name.split("_", 1)
        cached = existing_explanations.get(model_name, {}).get(str(model_number), {})

        for mol in mols:
            mol_id = mol.idx

            if cached and mol_id in cached and algorithm.value in cached[mol_id]:
                mask = cached[mol_id][algorithm.value]
                logger.debug(f"Using cached attribution for mol '{mol_id}', {model_log_name}.")
            else:
                logger.debug(f"Computing attribution for mol '{mol_id}', {model_log_name}.")
                mask = (
                    explainer(
                        x=mol.x,
                        edge_index=mol.edge_index,
                        batch_index=None,
                        edge_attr=mol.edge_attr,
                        monomer_weight=mol.weight_monomer,
                        index=0,
                    )
                    .node_mask.detach()
                    .numpy()
                    .tolist()
                )

            (
                node_masks.setdefault(model_name, {})
                .setdefault(model_number, {})
                .setdefault(mol_id, {})
            )[algorithm.value] = mask

    return node_masks


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
