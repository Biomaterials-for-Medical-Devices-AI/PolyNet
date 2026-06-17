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

from rdkit import Chem
import torch
import torch.nn.functional as F

from polynet.config.enums import (
    ExplainAlgorithm,
    FragmentationMethod,
    ImportanceNormalisationMethod,
    ProblemType,
)
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
    node_masks: dict,
    model_log_names: list[str],
    target_class: int | None,
    fragmentation_method: FragmentationMethod | str,
    normalisation_type: ImportanceNormalisationMethod,
) -> dict[str, list[float]]:
    """
    Collect all fragment attribution scores across every model and molecule
    into a per-fragment distribution ready for plotting.

    Unlike ``merge_fragment_attributions`` (which averages across models),
    every individual score is preserved so the resulting distribution reflects
    both model uncertainty and molecule variability.

    Normalisation is applied at the **(model × mol_id)** grain for Local, or
    once across all scores for Global, before flattening.

    Parameters
    ----------
    node_masks:
        Raw output of ``calculate_masking_attributions``:
        ``{model_name: {model_number: {mol_id: {alg_key: {ck: {monomer: {fk: {frag: [scores]}}}}}}}}``
    model_log_names:
        Ordered list of ``"{model_name}_{number}"`` strings — defines iteration order.
    target_class:
        The class to extract. Pass ``None`` for regression.
    fragmentation_method:
        The fragmentation strategy whose results to extract.
    normalisation_type:
        ``Local``          — each (model × mol_id) unit is independently scaled to [-1, 1].
        ``Global``         — all scores divided by the single largest absolute value found.
        ``NoNormalisation``— raw scores collected unchanged.

    Returns
    -------
    dict[str, list[float]]
        ``{frag_smiles: [s_model1_mol1_occ1, s_model1_mol1_occ2, ..., s_modelN_molM_occK, ...]}``.
        Every (model × mol_id × occurrence) triple contributes one data point,
        giving the full distribution for the ridge / violin plot.
    """
    fk = (
        fragmentation_method.value
        if isinstance(fragmentation_method, FragmentationMethod)
        else str(fragmentation_method)
    )
    ck = _class_key(target_class)

    def _unit_scores(mol_data: dict) -> dict[str, list[float]]:
        """Flatten all fragment scores for one (model × mol_id) unit into {frag: [scores]}."""
        unit: dict[str, list[float]] = {}
        for mon_data in mol_data.get(_ALGORITHM_KEY, {}).get(ck, {}).values():
            for frag_smiles, scores in mon_data.get(fk, {}).items():
                unit.setdefault(frag_smiles, []).extend(scores)
        return unit

    # --- Pass 1 (Global / PerModel): pre-compute divisors that require a full scan ---
    global_divisor: float = 1.0
    per_model_divisors: dict[str, float] = {}  # key: "{model_name}_{number}"

    if normalisation_type == ImportanceNormalisationMethod.Global:
        global_max = 0.0
        for log_name in model_log_names:
            model_name, number = log_name.split("_", 1)
            for mol_data in node_masks.get(model_name, {}).get(number, {}).values():
                for scores in _unit_scores(mol_data).values():
                    for s in scores:
                        if abs(s) > global_max:
                            global_max = abs(s)
        global_divisor = global_max or 1.0

    elif normalisation_type == ImportanceNormalisationMethod.PerModel:
        for log_name in model_log_names:
            model_name, number = log_name.split("_", 1)
            model_max = 0.0
            for mol_data in node_masks.get(model_name, {}).get(number, {}).values():
                for scores in _unit_scores(mol_data).values():
                    for s in scores:
                        if abs(s) > model_max:
                            model_max = abs(s)
            per_model_divisors[log_name] = model_max or 1.0

    # --- Pass 2: collect scores, normalising per unit ---
    distribution: dict[str, list[float]] = {}

    for log_name in model_log_names:
        model_name, number = log_name.split("_", 1)
        for mol_data in node_masks.get(model_name, {}).get(number, {}).values():
            unit = _unit_scores(mol_data)
            if not unit:
                continue

            if normalisation_type == ImportanceNormalisationMethod.Local:
                divisor = (
                    max((abs(s) for scores in unit.values() for s in scores), default=0.0) or 1.0
                )
            elif normalisation_type == ImportanceNormalisationMethod.Global:
                divisor = global_divisor
            elif normalisation_type == ImportanceNormalisationMethod.PerModel:
                divisor = per_model_divisors[log_name]
            else:
                divisor = 1.0

            for frag_smiles, scores in unit.items():
                distribution.setdefault(frag_smiles, []).extend(s / divisor for s in scores)

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
    monomer_id = getattr(mol, "monomer_id", None)

    with torch.no_grad():
        # --- Baseline prediction ---
        y_full = _get_scalar_prediction(
            model.forward(
                x=mol.x,
                edge_index=mol.edge_index,
                batch_index=None,
                edge_attr=mol.edge_attr,
                monomer_weight=monomer_weight,
                monomer_id=monomer_id,
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

        # Per-node monomer ids let us locate each monomer's graph nodes exactly,
        # even when wildcard ('*') atoms were stripped during featurisation
        # (IsAttachmentPoint), so the graph has fewer nodes than the RDKit mol.
        monomer_id_attr = getattr(mol, "monomer_id", None)
        node_monomer_ids = (
            monomer_id_attr.view(-1).tolist() if monomer_id_attr is not None else None
        )

        # Result keyed by monomer SMILES, then by fragment SMILES → list of
        # per-occurrence attributions (one entry per structural match)
        frag_attributions: dict[str, dict[str, list[float]]] = {}

        for monomer_idx, smiles in enumerate(mol.mols):
            rdkit_mol = Chem.MolFromSmiles(smiles)
            if rdkit_mol is None:
                logger.warning(f"Could not parse SMILES '{smiles}' in mol '{mol.idx}'. Skipping.")
                continue

            # Graph node positions for this monomer (contiguous in graph order).
            if node_monomer_ids is not None:
                node_positions = [i for i, mid in enumerate(node_monomer_ids) if mid == monomer_idx]
            else:
                # Fallback for graphs without monomer_id: assume monomers are laid
                # out by full RDKit atom count (no wildcard stripping).
                start = sum(Chem.MolFromSmiles(s).GetNumAtoms() for s in mol.mols[:monomer_idx])
                node_positions = list(range(start, start + rdkit_mol.GetNumAtoms()))

            if not node_positions:
                continue

            atom_offset = node_positions[0]
            n_graph_nodes = len(node_positions)

            # Map RDKit atom index (with dummies) → graph-local node index. Detect
            # whether wildcards were stripped by comparing the node and atom counts.
            full_n = rdkit_mol.GetNumAtoms()
            dummy_idxs = {a.GetIdx() for a in rdkit_mol.GetAtoms() if a.GetAtomicNum() == 0}
            if n_graph_nodes == full_n - len(dummy_idxs):
                # Wildcards stripped: non-dummy atoms kept in original order.
                kept = [i for i in range(full_n) if i not in dummy_idxs]
                atom_to_node = {orig: new for new, orig in enumerate(kept)}
            elif n_graph_nodes == full_n:
                atom_to_node = {i: i for i in range(full_n)}
            else:
                logger.warning(
                    f"Atom/graph-node count mismatch for monomer '{smiles}' in mol "
                    f"'{mol.idx}' ({n_graph_nodes} graph nodes vs {full_n} atoms, "
                    f"{len(dummy_idxs)} wildcard(s)). Skipping this monomer."
                )
                continue

            frags = fragment_and_match(smiles, fragmentation_method)
            monomer_frags: dict[str, list[float]] = {}

            for frag_smiles, atom_indices_list in frags.items():
                occurrence_scores: list[float] = []

                for atom_indices in atom_indices_list:
                    # Translate to graph-local node indices, dropping wildcard atoms
                    # that have no corresponding graph node.
                    local_nodes = [atom_to_node[i] for i in atom_indices if i in atom_to_node]
                    if not local_nodes:
                        continue

                    # Remove this occurrence's nodes from the pooling operation.
                    # Zeroing would dilute mean pooling by keeping the node count
                    # unchanged; deletion ensures the pooling denominator (for
                    # mean) and the candidate set (for max) reflect only the
                    # nodes that remain.
                    global_indices = [atom_offset + j for j in local_nodes]
                    keep = torch.ones(h.shape[0], dtype=torch.bool)
                    keep[global_indices] = False

                    # If the fragment spans every node in the graph (e.g. a small
                    # monomer with no breakable bonds is returned whole), masking
                    # it leaves nothing to pool and the masked prediction is
                    # undefined. Skip this occurrence rather than crash.
                    if not bool(keep.any()):
                        logger.debug(
                            f"Fragment '{frag_smiles}' covers all graph nodes of mol "
                            f"'{mol.idx}'; skipping (no nodes left to pool)."
                        )
                        continue

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

                if occurrence_scores:
                    monomer_frags[frag_smiles] = occurrence_scores

            if monomer_frags:
                frag_attributions[smiles] = monomer_frags

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
