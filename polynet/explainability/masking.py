"""
polynet.explainability.masking
================================
Chemistry-aware masking attribution for GNN explainability.

Implements the fragment-masking strategy from:
    Wellawatte et al., Nat. Commun. 14, 2023. https://doi.org/10.1038/s41467-023-38192-3

For each fragment found in a molecule, all atoms belonging to that fragment are
zeroed out in the pre-pooling node embedding space.  The attribution is defined as:

    attribution(fragment) = Y_pred_full − Y_pred_masked

Molecules that do not contain a given fragment produce no entry for that fragment
— they are not counted as zero.

Public API
----------
::

    from polynet.explainability.masking import (
        calculate_masking_attributions,
        merge_fragment_attributions,
        fragment_attributions_to_distribution,
    )
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from rdkit import Chem

from polynet.config.enums import ExplainAlgorithm, FragmentationMethod, ProblemType
from polynet.utils.chem_utils import fragment_and_match

logger = logging.getLogger(__name__)

_ALGORITHM_KEY = ExplainAlgorithm.ChemistryMasking.value


def _class_key(target_class: int | None) -> str:
    """Stable string key for the target class level of the result dict."""
    return "regression" if target_class is None else str(target_class)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_masking_attributions(
    mols: list,
    models: dict,
    fragmentation_method: FragmentationMethod | str,
    problem_type: ProblemType | str,
    existing_explanations: dict,
    target_class: int | None = None,
) -> dict:
    """
    Compute fragment-level masking attributions for each model × molecule pair.

    Parameters
    ----------
    mols:
        List of PyG graph objects to explain.
    models:
        Dict of ``{"{model_name}_{number}": model}`` — the trained GNN ensemble.
    fragmentation_method:
        Fragmentation strategy passed to ``fragment_and_match``.
    problem_type:
        Regression or classification — controls how the scalar prediction is extracted.
    existing_explanations:
        Previously cached results loaded from the explanation JSON file.
        Structure: ``{model_name: {model_number: {mol_id: {algorithm: result}}}}``.
    target_class:
        For classification problems, the class index whose predicted probability
        is used as the scalar to perturb. Must be provided when
        ``problem_type == ProblemType.Classification``.

    Returns
    -------
    dict
        ``{model_name: {model_number: {mol_id: {"chemistry_masking": {class_key: {monomer_smiles: {frag_key: {frag_smiles: [scores]}}}}}}}}``.
        ``class_key`` is ``str(target_class)`` for classification or ``"regression"``.
        ``frag_key`` is the string value of ``fragmentation_method`` (e.g. ``"brics"``).
        Molecules with no matching fragments produce an empty inner dict for their class key.
    """
    fragmentation_method = (
        FragmentationMethod(fragmentation_method)
        if isinstance(fragmentation_method, str)
        else fragmentation_method
    )
    problem_type = ProblemType(problem_type) if isinstance(problem_type, str) else problem_type

    fk = fragmentation_method.value
    ck = _class_key(target_class)
    result: dict = {}

    for model_log_name, model in models.items():
        model_name, model_number = model_log_name.split("_", 1)
        cached = existing_explanations.get(model_name, {}).get(str(model_number), {})

        model.eval()

        for mol in mols:
            mol_id = mol.idx

            # Cache hit: ck exists AND every monomer already has this fk computed
            cached_ck = cached.get(mol_id, {}).get(_ALGORITHM_KEY, {}).get(ck, {})
            already_cached = bool(cached_ck) and all(
                fk in mon_data for mon_data in cached_ck.values()
            )

            if already_cached:
                # Reconstruct {monomer_smiles: {frag_smiles: [scores]}} from stored format
                raw_attributions = {
                    monomer: mon_data[fk]
                    for monomer, mon_data in cached_ck.items()
                    if fk in mon_data
                }
                logger.debug(
                    f"Using cached masking attribution for mol '{mol_id}', "
                    f"{model_log_name}, fragmentation '{fk}', class '{ck}'."
                )
            else:
                logger.debug(
                    f"Computing masking attribution for mol '{mol_id}', "
                    f"{model_log_name}, fragmentation '{fk}', class '{ck}'."
                )
                raw_attributions = _compute_masking_attributions(
                    mol=mol,
                    model=model,
                    fragmentation_method=fragmentation_method,
                    problem_type=problem_type,
                    target_class=target_class,
                )

            # Store with fk wrapping: algorithm → class → monomer → frag_key → frags
            mol_entry = (
                result.setdefault(model_name, {})
                .setdefault(model_number, {})
                .setdefault(mol_id, {})
                .setdefault(_ALGORITHM_KEY, {})
                .setdefault(ck, {})
            )
            for monomer_smiles, frag_dict in raw_attributions.items():
                mol_entry.setdefault(monomer_smiles, {})[fk] = frag_dict

    return result


def merge_fragment_attributions(node_masks: dict, model_log_names: list[str]) -> dict:
    """
    Average fragment attribution dicts across ensemble models.

    Only models that produced a score for a given (monomer, fragment) pair are
    counted in the average — absent pairs are excluded from the denominator.

    Parameters
    ----------
    node_masks:
        Structure:
        ``{model_name: {model_number: {mol_id: {"chemistry_masking": {class_key: {monomer_smiles: {frag_key: {frag: [scores]}}}}}}}}``.
    model_log_names:
        List of model log names (``"{model_name}_{number}"``).

    Returns
    -------
    dict
        ``{mol_id: {"chemistry_masking": {class_key: {monomer_smiles: {frag_key: {frag_smiles: [avg_score_per_occurrence]}}}}}}``.
        Lists are averaged element-wise across models — occurrence order is
        preserved since fragmentation is deterministic for the same SMILES.
    """
    # accumulator[mol_id][class_key][monomer_smiles][frag_key][frag_smiles] = running element-wise sum
    accumulator: dict = {}
    counts: dict = {}

    for log_name in model_log_names:
        model_name, number = log_name.split("_", 1)
        model_dict = node_masks.get(model_name, {}).get(number, {})

        for mol_id, algos in model_dict.items():
            for ck, mon_fk_dict in algos.get(_ALGORITHM_KEY, {}).items():
                for monomer_smiles, fk_dict in mon_fk_dict.items():
                    for fk, frag_dict in fk_dict.items():
                        mon_acc = (
                            accumulator.setdefault(mol_id, {})
                            .setdefault(ck, {})
                            .setdefault(monomer_smiles, {})
                            .setdefault(fk, {})
                        )
                        mon_cnt = (
                            counts.setdefault(mol_id, {})
                            .setdefault(ck, {})
                            .setdefault(monomer_smiles, {})
                            .setdefault(fk, {})
                        )
                        for frag_smiles, scores in frag_dict.items():
                            existing = mon_acc.get(frag_smiles)
                            if existing is None:
                                mon_acc[frag_smiles] = list(scores)
                            else:
                                mon_acc[frag_smiles] = [a + b for a, b in zip(existing, scores)]
                            mon_cnt[frag_smiles] = mon_cnt.get(frag_smiles, 0) + 1

    return {
        mol_id: {
            _ALGORITHM_KEY: {
                ck: {
                    monomer_smiles: {
                        fk: {
                            frag: [
                                s / counts[mol_id][ck][monomer_smiles][fk][frag]
                                for s in accumulator[mol_id][ck][monomer_smiles][fk][frag]
                            ]
                            for frag in accumulator[mol_id][ck][monomer_smiles][fk]
                        }
                        for fk in accumulator[mol_id][ck][monomer_smiles]
                    }
                    for monomer_smiles in accumulator[mol_id][ck]
                }
                for ck in accumulator[mol_id]
            }
        }
        for mol_id in accumulator
    }


def fragment_attributions_to_distribution(
    merged: dict,
    target_class: int | None,
    fragmentation_method: FragmentationMethod | str,
) -> dict[str, list[float]]:
    """
    Flatten per-molecule, per-monomer attribution dicts to a per-fragment list
    for a specific target class and fragmentation method.

    Suitable for passing directly to ``plot_attribution_distribution``.
    Monomers that do not contain a fragment are automatically excluded.

    Parameters
    ----------
    merged:
        Output of ``merge_fragment_attributions``:
        ``{mol_id: {"chemistry_masking": {class_key: {monomer_smiles: {frag_key: {frag_smiles: [scores]}}}}}}``.
    target_class:
        The class whose attributions to extract. Pass ``None`` for regression.
    fragmentation_method:
        The fragmentation strategy whose results to extract.

    Returns
    -------
    dict[str, list[float]]
        ``{frag_smiles: [score_occ1_mol1, score_occ2_mol1, score_occ1_mol2, ...]}``.
        Each entry corresponds to one structural occurrence across all molecules
        and monomers — suitable for a distribution/violin plot.
    """
    ck = _class_key(target_class)
    fk = (
        fragmentation_method.value
        if isinstance(fragmentation_method, FragmentationMethod)
        else str(fragmentation_method)
    )
    distribution: dict[str, list[float]] = {}

    for mol_id, algos in merged.items():
        for monomer_smiles, fk_dict in algos.get(_ALGORITHM_KEY, {}).get(ck, {}).items():
            for frag_smiles, scores in fk_dict.get(fk, {}).items():
                distribution.setdefault(frag_smiles, []).extend(scores)

    return distribution


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_masking_attributions(
    mol,
    model,
    fragmentation_method: FragmentationMethod,
    problem_type: ProblemType,
    target_class: int | None,
) -> dict[str, float]:
    """
    Core masking computation for a single molecule against a single model.

    For each fragment found in the molecule, ALL occurrences are masked
    simultaneously (their pre-pooling node embeddings are zeroed out).
    Attribution is Y_pred_full − Y_pred_masked.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{monomer_smiles: {frag_smiles: attribution_score}}``.
        Empty dict if no fragments are matched in any monomer.
    """
    polymer_descriptors = getattr(mol, "polymer_descriptors", None)
    monomer_weight = getattr(mol, "weight_monomer", None)

    with torch.no_grad():
        # --- Baseline prediction ---
        y_full = _get_scalar_prediction(
            model.forward(
                x=mol.x,
                edge_index=mol.edge_index,
                batch_index=None,
                edge_attr=mol.edge_attr,
                monomer_weight=monomer_weight,
                polymer_descriptors=polymer_descriptors,
            ),
            problem_type=problem_type,
            target_class=target_class,
        )

        # --- Pre-pooling node embeddings ---
        h = model.get_node_embeddings(
            x=mol.x,
            edge_index=mol.edge_index,
            batch_index=None,
            edge_attr=mol.edge_attr,
            monomer_weight=monomer_weight,
        )

        # batch_index required by PyG pooling functions; single graph → all zeros
        batch_idx = torch.zeros(h.shape[0], dtype=torch.long)

        # Result keyed by monomer SMILES, then by fragment SMILES → list of
        # per-occurrence attributions (one entry per structural match)
        frag_attributions: dict[str, dict[str, list[float]]] = {}
        atom_offset = 0
        old_smiles = ""
        for smiles in mol.mols:
            if smiles == old_smiles:
                continue

            rdkit_mol = Chem.MolFromSmiles(smiles)
            if rdkit_mol is None:
                logger.warning(f"Could not parse SMILES '{smiles}' in mol '{mol.idx}'. Skipping.")
                continue

            frags = fragment_and_match(smiles, fragmentation_method)
            monomer_frags: dict[str, list[float]] = {}

            for frag_smiles, atom_indices_list in frags.items():
                occurrence_scores: list[float] = []

                for atom_indices in atom_indices_list:
                    # Remove this occurrence's nodes from the pooling operation.
                    # Zeroing would dilute mean pooling by keeping the node count
                    # unchanged; deletion ensures the pooling denominator (for
                    # mean) and the candidate set (for max) reflect only the
                    # nodes that remain.
                    global_indices = [atom_offset + i for i in atom_indices]
                    keep = torch.ones(h.shape[0], dtype=torch.bool)
                    keep[global_indices] = False

                    pooled = model.pooling_fn(h[keep], batch_idx[keep])

                    # Polymer descriptors are concatenated to the pooled embedding
                    # before the readout MLP (mirrors BaseNetwork.forward)
                    if polymer_descriptors is not None and model.n_polymer_descriptors > 0:
                        pooled = torch.cat([pooled, polymer_descriptors], dim=1)

                    y_masked = _get_scalar_prediction(
                        model.readout_function(pooled),
                        problem_type=problem_type,
                        target_class=target_class,
                    )

                    occurrence_scores.append(float(y_full - y_masked))

                monomer_frags[frag_smiles] = occurrence_scores

            if monomer_frags:
                frag_attributions[smiles] = monomer_frags

            atom_offset += rdkit_mol.GetNumAtoms()
            old_smiles = smiles

    return frag_attributions


def _get_scalar_prediction(
    output: torch.Tensor, problem_type: ProblemType, target_class: int | None
) -> float:
    """
    Extract a single scalar from a model's raw output tensor.

    For regression: the raw output value.
    For classification: the softmax probability of ``target_class``.
    """
    if problem_type == ProblemType.Regression:
        return output.reshape(-1)[0].item()
    else:
        probs = F.softmax(output, dim=-1)
        return probs[0, target_class].item()
