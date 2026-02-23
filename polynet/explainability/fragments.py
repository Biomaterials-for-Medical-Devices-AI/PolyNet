"""
polynet.explainability.fragments
=================================
Fragment-level attribution aggregation for polymer GNN explainability.

Maps per-atom attribution scores from ``merge_attribution_masks`` onto
molecular fragments produced by a chosen fragmentation strategy, enabling
interpretability at the level of functional groups and substructures
rather than individual atoms.

Public API
----------
::

    from polynet.explainability.fragments import get_fragment_importance
"""

from __future__ import annotations

import logging

import numpy as np
from rdkit import Chem

from polynet.config.enums import ExplainAlgorithm, FragmentationMethod
from polynet.utils.chem_utils import fragment_and_match

logger = logging.getLogger(__name__)


def get_fragment_importance(
    mols: list,
    node_masks: dict,
    algorithm: ExplainAlgorithm | str,
    fragmentation_method: FragmentationMethod | str,
) -> dict[str, list[float]]:
    """
    Aggregate per-atom attributions to fragment-level importance scores.

    For each molecule in ``mols``, fragments each constituent monomer SMILES
    using the given fragmentation strategy, then sums the attributions of
    all atoms belonging to each fragment. Fragments shorter than 3 characters
    (trivial groups) are skipped.

    Parameters
    ----------
    mols:
        List of PyG graph objects to explain. Each ``mol.mols`` attribute
        should contain the list of constituent monomer SMILES strings.
    node_masks:
        Merged attribution masks as returned by ``merge_attribution_masks``.
        Structure: ``{mol_id: {algorithm: mask_array}}``.
    algorithm:
        The attribution algorithm whose masks to aggregate.
    fragmentation_approach:
        Fragmentation strategy passed to ``fragment_and_match``.

    Returns
    -------
    dict[str, list[float]]
        Mapping from fragment SMILES to a list of per-occurrence importance
        scores (one score per fragment match across all molecules). Suitable
        for passing directly to ``plot_attribution_distribution``.

    Notes
    -----
    Atom indices in multi-monomer polymers are offset by the cumulative
    atom count of preceding monomers, so that indices correctly address
    the flattened ``node_mask`` array.
    """
    algorithm = ExplainAlgorithm(algorithm) if isinstance(algorithm, str) else algorithm

    frags_importances: dict[str, list[float]] = {}

    for mol in mols:
        mol_idx = mol.idx
        frag_importance = np.asarray(node_masks[mol_idx][algorithm.value])
        atom_offset = 0

        for smiles in mol.mols:
            rdkit_mol = Chem.MolFromSmiles(smiles)
            if rdkit_mol is None:
                logger.warning(f"Could not parse SMILES '{smiles}' in mol '{mol_idx}'. Skipping.")
                atom_offset += 0
                continue

            frags = fragment_and_match(smiles, fragmentation_method)

            for frag_smiles, atom_indices_list in frags.items():
                if len(frag_smiles) < 3:
                    continue  # skip trivial groups

                if frag_smiles not in frags_importances:
                    frags_importances[frag_smiles] = []

                for atom_indices in atom_indices_list:
                    global_indices = [atom_offset + i for i in atom_indices]
                    frag_score = float(np.sum(frag_importance[global_indices]))
                    frags_importances[frag_smiles].append(frag_score)

            atom_offset += rdkit_mol.GetNumAtoms()

    return frags_importances
