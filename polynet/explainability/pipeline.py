"""
polynet.explainability.pipeline
=================================
Pure computation pipeline for GNN attribution explanation.

``run_explanation`` orchestrates the full explanation workflow:
building explainers, computing attributions, merging masks, slicing
to a feature, aggregating fragment importances, normalising weights,
and splitting per-monomer. It returns plain data structures with no
Streamlit dependency — all rendering is handled by the calling app layer.

Public API
----------
::

    from polynet.explainability.pipeline import run_explanation, ExplanationResult
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rdkit import Chem

from polynet.config.enums import (
    ExplainAlgorithm,
    FragmentationMethod,
    ImportanceNormalisationMethod,
    ProblemType,
)
from polynet.explainability.attributions import (
    build_explainers,
    calculate_attributions,
    deep_update,
    merge_attribution_masks,
    slice_masks_to_feature,
)
from polynet.explainability.fragments import get_fragment_importance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class MoleculeExplanation:
    """
    Explanation result for a single polymer molecule.

    Attributes
    ----------
    mol_id:
        Sample identifier.
    monomer_smiles:
        List of constituent monomer SMILES strings.
    per_monomer_weights:
        Per-atom weights split by monomer, each normalised and thresholded.
        Shape: one list per monomer, each list has length ``n_atoms_in_monomer``.
    raw_mask:
        Full per-atom attribution vector (summed over feature dimension)
        before normalisation.
    """

    mol_id: str
    monomer_smiles: list[str]
    per_monomer_weights: list[list[float]]
    raw_mask: list[float]


@dataclass
class ExplanationResult:
    """
    Full result of ``run_explanation``.

    Attributes
    ----------
    mol_explanations:
        Per-molecule results for the molecules selected for plotting.
    fragment_importances:
        Fragment SMILES → list of per-occurrence importance scores.
    algorithm:
        The attribution algorithm used.
    """

    mol_explanations: list[MoleculeExplanation]
    fragment_importances: dict[str, list[float]]
    algorithm: ExplainAlgorithm


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_explanation(
    models: dict,
    dataset,
    explain_mol_ids: list,
    plot_mol_ids: list,
    algorithm: ExplainAlgorithm | str,
    problem_type: ProblemType | str,
    experiment_path: Path,
    node_features: dict,
    explain_feature: str = "All Features",
    normalisation: ImportanceNormalisationMethod | str = ImportanceNormalisationMethod.Local,
    cutoff: float = 0.1,
    fragmentation_approach: FragmentationMethod | str = FragmentationMethod.BRICS,
) -> ExplanationResult:
    """
    Compute GNN attributions and return structured explanation results.

    Workflow
    --------
    1. Build one ``Explainer`` per trained model.
    2. Load attribution cache from disk; compute only missing entries.
    3. Merge and average masks across models.
    4. Optionally slice masks to a single node feature.
    5. Aggregate per-atom attributions to fragment-level scores.
    6. Normalise atom weights (local or global) and apply cutoff.
    7. Split per-atom weights back into per-monomer lists.

    Parameters
    ----------
    models:
        Dict of ``{"{arch}_{iteration}": model}`` as from ``train_gnn_ensemble``.
    dataset:
        Full PyG dataset (used to filter molecules by ID).
    explain_mol_ids:
        Sample IDs to compute attributions for (superset of ``plot_mol_ids``).
    plot_mol_ids:
        Sample IDs to include in the final ``MoleculeExplanation`` results.
    algorithm:
        Attribution algorithm to use.
    problem_type:
        Classification or regression.
    experiment_path:
        Root experiment path; attributions are cached under
        ``{experiment_path}/explanations/explanations.json``.
    node_features:
        Node feature configuration dict from the experiment config.
    explain_feature:
        Feature name to slice masks to. Pass ``"All Features"`` to use
        the full feature vector.
    normalisation:
        ``Local`` — each molecule normalised by its own maximum.
        ``Global`` — all molecules normalised by the global maximum.
        ``None`` — no normalisation.
    cutoff:
        Absolute attribution threshold below which atom weights are
        zeroed out (applied after normalisation).
    fragmentation_approach:
        Fragmentation strategy for ``get_fragment_importance``.

    Returns
    -------
    ExplanationResult
        Structured result containing per-molecule explanations and
        fragment importances. Pass to app layer for Streamlit rendering.
    """
    from polynet.app.options.file_paths import (
        explanation_json_file_path,
        explanation_parent_directory,
    )
    from polynet.training.gnn import filter_dataset_by_ids

    algorithm = ExplainAlgorithm(algorithm) if isinstance(algorithm, str) else algorithm
    problem_type = ProblemType(problem_type) if isinstance(problem_type, str) else problem_type
    normalisation = (
        ImportanceNormalisationMethod(normalisation)
        if isinstance(normalisation, str)
        else normalisation
    )

    # --- Explainers ---
    explainers = build_explainers(models, algorithm, problem_type)

    # --- Attribution cache ---
    explain_path = explanation_parent_directory(experiment_path)
    explain_path.mkdir(parents=True, exist_ok=True)
    cache_file = explanation_json_file_path(experiment_path)

    existing: dict = {}
    if cache_file.exists():
        with open(cache_file) as f:
            existing = json.load(f)

    mols = filter_dataset_by_ids(dataset, explain_mol_ids)

    # --- Compute attributions ---
    node_masks = calculate_attributions(
        mols=mols, existing_explanations=existing, algorithm=algorithm, explainers=explainers
    )

    # Persist updated cache
    merged_cache = deep_update(existing, node_masks)
    with open(cache_file, "w") as f:
        json.dump(merged_cache, f, indent=4)

    # --- Merge across models ---
    merged_masks = merge_attribution_masks(node_masks, list(models.keys()))

    # --- Optional feature slicing ---
    merged_masks = slice_masks_to_feature(
        node_masks=merged_masks,
        mols=mols,
        algorithm=algorithm,
        node_features=node_features,
        feature_name=explain_feature,
    )

    # --- Fragment importance ---
    fragment_importances = get_fragment_importance(
        mols=mols,
        node_masks=merged_masks,
        algorithm=algorithm,
        fragmentation_approach=fragmentation_approach,
    )

    # --- Per-atom weight normalisation ---
    # Compute global max before iterating if needed
    global_max: float | None = None
    if normalisation == ImportanceNormalisationMethod.Global:
        global_max = 0.0
        for mol in mols:
            raw = np.array(merged_masks[mol.idx][algorithm.value]).sum(axis=1)
            local_max = float(np.max(np.abs(raw)))
            if local_max > global_max:
                global_max = local_max
                logger.debug(f"New global max: {global_max:.4f} from mol '{mol.idx}'.")

    # --- Build per-molecule results ---
    plot_mol_set = set(plot_mol_ids)
    plot_mols = [m for m in mols if m.idx in plot_mol_set]
    mol_explanations: list[MoleculeExplanation] = []

    for mol in plot_mols:
        raw_mask = np.array(merged_masks[mol.idx][algorithm.value]).sum(axis=1)

        if normalisation == ImportanceNormalisationMethod.Local:
            local_max = float(np.max(np.abs(raw_mask)))
            weights = raw_mask / local_max if local_max > 0 else raw_mask
        elif normalisation == ImportanceNormalisationMethod.Global and global_max:
            weights = raw_mask / global_max if global_max > 0 else raw_mask
        else:
            weights = raw_mask.copy()

        weights = np.where(np.abs(weights) > cutoff, weights, 0.0)

        # Split flattened weights back into per-monomer lists
        per_monomer: list[list[float]] = []
        atom_offset = 0
        for smiles in mol.mols:
            rdmol = Chem.MolFromSmiles(smiles)
            n = rdmol.GetNumAtoms() if rdmol else 0
            per_monomer.append(weights[atom_offset : atom_offset + n].tolist())
            atom_offset += n

        mol_explanations.append(
            MoleculeExplanation(
                mol_id=mol.idx,
                monomer_smiles=list(mol.mols),
                per_monomer_weights=per_monomer,
                raw_mask=raw_mask.tolist(),
            )
        )

    return ExplanationResult(
        mol_explanations=mol_explanations,
        fragment_importances=fragment_importances,
        algorithm=algorithm,
    )
