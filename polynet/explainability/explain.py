"""
polynet.explainability.explain
================================
High-level XAI pipeline for chemistry-masking fragment attribution.

These functions form the public computation API — they have **no Streamlit
dependency** and can be called from notebooks, scripts, or any other frontend.

Typical notebook usage::

    from polynet.explainability import (
        compute_and_cache_masking,
        build_display_data,
        compute_global_attribution,
        compute_local_attribution,
    )
    from polynet.config.enums import (
        FragmentationMethod,
        ImportanceNormalisationMethod,
        AttributionPlotType,
        ProblemType,
    )

    combined = compute_and_cache_masking(
        models, experiment_path, dataset, mol_ids,
        ProblemType.Regression, FragmentationMethod.BRICS,
    )
    result = compute_global_attribution(
        models, experiment_path, dataset, mol_ids,
        ProblemType.Regression, top_n=10,
    )
    result.figure.savefig("attribution_ridge.png")

    for mol_result in compute_local_attribution(
        models, experiment_path, dataset, mol_ids,
        ProblemType.Regression,
    ):
        mol_result.attribution_df.to_csv(f"{mol_result.mol_idx}_attribution.csv")
        mol_result.mol_figure.savefig(f"{mol_result.mol_idx}_heatmap.png")
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem

from polynet.config.constants import ResultColumn
from polynet.config.enums import (
    AttributionPlotType,
    ExplainAlgorithm,
    FragmentationMethod,
    ImportanceNormalisationMethod,
    ProblemType,
)
from polynet.config.paths import explanation_json_file_path, explanation_parent_directory
from polynet.explainability.attributions import deep_update
from polynet.explainability.masking import (
    calculate_masking_attributions,
    fragment_attributions_to_distribution,
    merge_fragment_attributions,
)
from polynet.explainability.visualization import (
    get_cmap,
    plot_attribution_bar,
    plot_attribution_distribution,
    plot_attribution_strip,
    plot_mols_with_weights,
)
from polynet.utils import filter_dataset_by_ids
from polynet.utils.chem_utils import fragment_and_match

logger = logging.getLogger(__name__)

_ALG_KEY = ExplainAlgorithm.ChemistryMasking.value


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GlobalAttributionResult:
    """
    Return value of :func:`compute_global_attribution`.

    Attributes
    ----------
    figure:
        Matplotlib figure (Ridge / Bar / Strip plot).  ``None`` when no
        fragment attributions were found (check ``warning``).
    attribution_dict:
        ``{frag_smiles: [scores]}`` — the raw distribution passed to the plot.
    n_mols:
        Number of molecules included in the plot.
    n_models:
        Number of model instances included.
    n_frags_total:
        Total unique fragments found across all selected molecules and models.
    n_shown:
        Number of fragments shown (top + bottom combined, after ``top_n`` filtering).
    target_class:
        The classification class index used, or ``None`` for regression.
    normalisation_type:
        The normalisation strategy applied.
    warning:
        Non-empty string if no attributions were found; ``None`` otherwise.
    """

    figure: plt.Figure | None
    attribution_dict: dict[str, list[float]]
    n_mols: int
    n_models: int
    n_frags_total: int
    n_shown: int
    target_class: int | None
    normalisation_type: ImportanceNormalisationMethod
    warning: str | None = None


@dataclass
class MolAttributionResult:
    """
    Return value for a single molecule from :func:`compute_local_attribution`.

    Attributes
    ----------
    mol_idx:
        The molecule identifier (matches the dataset index).
    info_msg:
        Human-readable summary of the attribution run for this molecule.
    true_label:
        String representation of the ground-truth label, or ``"N/A"``.
    predicted_label:
        String representation of the model's predicted label, or ``"N/A"``.
    attribution_df:
        Per-fragment attribution table (empty DataFrame if ``warning`` is set).
    mol_figure:
        Atom-level heatmap figure.  ``None`` if ``warning`` is set.
    warning:
        Non-empty string if no fragments were matched; ``None`` otherwise.
    """

    mol_idx: str | int
    info_msg: str
    true_label: str
    predicted_label: str
    attribution_df: pd.DataFrame
    mol_figure: plt.Figure | None
    warning: str | None = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _frag_key(fragmentation_approach) -> str:
    return (
        fragmentation_approach.value
        if isinstance(fragmentation_approach, FragmentationMethod)
        else str(fragmentation_approach)
    )


def _class_key(target_class: int | None) -> str:
    return "regression" if target_class is None else str(target_class)


# ---------------------------------------------------------------------------
# Core: cache management
# ---------------------------------------------------------------------------


def compute_and_cache_masking(
    models: dict,
    experiment_path: Path,
    dataset,
    explain_mols: list,
    problem_type: ProblemType,
    fragmentation_approach=FragmentationMethod.BRICS,
    target_class: int | None = None,
) -> dict:
    """
    Load the JSON cache, compute any missing masking attributions, persist,
    and return the full combined explanation dict.

    Raw scores are always stored; normalisation is **never** written to the
    cache.  Call :func:`build_display_data` on the returned dict to obtain a
    view filtered to specific models and molecules.

    Parameters
    ----------
    models:
        ``{"{model_name}_{number}": model}`` — the trained GNN ensemble.
    experiment_path:
        Root path of the PolyNet experiment directory.
    dataset:
        The PyG dataset (``CustomPolymerGraph``) to draw molecules from.
    explain_mols:
        List of molecule IDs to explain.
    problem_type:
        Regression or classification.
    fragmentation_approach:
        Fragmentation strategy (default: BRICS).
    target_class:
        For classification, the class index to attribute.

    Returns
    -------
    dict
        Full combined explanation dict (all models, all cached molecules).
    """
    explain_path = explanation_parent_directory(experiment_path)
    if not explain_path.exists():
        explain_path.mkdir(parents=True, exist_ok=True)

    explanation_file = explanation_json_file_path(experiment_path=experiment_path)
    if explanation_file.exists():
        with open(explanation_file) as f:
            existing_explanations = json.load(f)
    else:
        existing_explanations = {}

    if isinstance(explain_mols, str):
        explain_mols = [explain_mols]

    mols = filter_dataset_by_ids(dataset, explain_mols)

    node_masks = calculate_masking_attributions(
        mols=mols,
        models=models,
        fragmentation_method=fragmentation_approach,
        problem_type=problem_type,
        existing_explanations=existing_explanations,
        target_class=target_class,
    )

    combined_explanations = deep_update(existing_explanations, node_masks)
    with open(explanation_file, "w") as f:
        json.dump(combined_explanations, f, indent=4)

    return combined_explanations


def build_display_data(combined_explanations: dict, models: dict, mol_ids: list) -> dict:
    """
    Filter the full cache to only the requested models and molecule IDs.

    JSON keys are always strings; mol_ids are str-normalised for the lookup
    so integer IDs from pandas indices match correctly.

    Parameters
    ----------
    combined_explanations:
        Full explanation cache as returned by :func:`compute_and_cache_masking`.
    models:
        ``{"{model_name}_{number}": model}`` — only these model instances are kept.
    mol_ids:
        Only molecules whose ID appears in this list are included.

    Returns
    -------
    dict
        Same nested structure as the cache, restricted to the requested subset.
    """
    mol_id_set = {str(m) for m in mol_ids}
    display_data: dict = {}
    for model_log_name in models.keys():
        model_name, model_number = model_log_name.split("_", 1)
        mol_cache = combined_explanations.get(model_name, {}).get(model_number, {})
        for mol_id, mol_entry in mol_cache.items():
            if str(mol_id) in mol_id_set:
                (display_data.setdefault(model_name, {}).setdefault(model_number, {}))[
                    mol_id
                ] = mol_entry
    return display_data


# ---------------------------------------------------------------------------
# Core: atom-weight mapping
# ---------------------------------------------------------------------------


def fragment_attributions_to_atom_weights(
    mol, monomer_dict: dict, frag_key: str, fragmentation_approach
) -> list[list[float]]:
    """
    Map per-fragment masking attributions back to per-atom weights for visualisation.

    Parameters
    ----------
    mol:
        A PyG graph object with ``.mols`` (list of monomer SMILES) attribute.
    monomer_dict:
        ``{monomer_smiles: {frag_key: {frag_smiles: [score_per_occurrence]}}}``.
    frag_key:
        String value of the fragmentation method (e.g. ``"brics"``).
    fragmentation_approach:
        Fragmentation method passed to ``fragment_and_match``.

    Returns
    -------
    list[list[float]]
        One list of per-atom weights per monomer SMILES in ``mol.mols``.
        Each inner list has length equal to the number of heavy atoms in
        the corresponding monomer.
    """
    weights_list = []
    for smiles in mol.mols:
        rdkit_mol = Chem.MolFromSmiles(smiles)
        if rdkit_mol is None:
            weights_list.append([])
            continue

        weights = [0.0] * rdkit_mol.GetNumAtoms()
        frag_scores = monomer_dict.get(smiles, {}).get(frag_key, {})
        if frag_scores:
            frags = fragment_and_match(smiles, fragmentation_approach)
            for frag_smiles, atom_indices_list in frags.items():
                scores = frag_scores.get(frag_smiles, [])
                for occ_idx, atom_indices in enumerate(atom_indices_list):
                    if occ_idx < len(scores):
                        score = scores[occ_idx]
                        for atom_idx in atom_indices:
                            weights[atom_idx] = score
        weights_list.append(weights)
    return weights_list


# ---------------------------------------------------------------------------
# High-level: global (population) attribution
# ---------------------------------------------------------------------------


def compute_global_attribution(
    models: dict,
    experiment_path: Path,
    dataset,
    explain_mols: list,
    problem_type: ProblemType,
    neg_color: str = "#40bcde",
    pos_color: str = "#e64747",
    normalisation_type: ImportanceNormalisationMethod = ImportanceNormalisationMethod.PerModel,
    fragmentation_approach=FragmentationMethod.BRICS,
    target_class: int | None = None,
    top_n: int | None = None,
    plot_type: AttributionPlotType = AttributionPlotType.Ridge,
) -> GlobalAttributionResult:
    """
    Compute the population-level fragment attribution plot.

    Covers all selected molecules and model instances, preserving every
    individual score so the distribution reflects both model uncertainty
    and molecule variability.

    Parameters
    ----------
    models:
        ``{"{model_name}_{number}": model}`` ensemble.
    experiment_path:
        Root path of the PolyNet experiment directory.
    dataset:
        The PyG dataset to draw molecules from.
    explain_mols:
        Molecule IDs to include in the distribution.
    problem_type:
        Regression or classification.
    neg_color:
        Hex colour for the negative-attribution end of the colormap.
    pos_color:
        Hex colour for the positive-attribution end.
    normalisation_type:
        Normalisation strategy (Local, Global, PerModel, NoNormalisation).
    fragmentation_approach:
        Fragmentation strategy (default: BRICS).
    target_class:
        For classification, the class index to attribute.
    top_n:
        If set, show only the top-N and bottom-N fragments by mean attribution.
    plot_type:
        ``Ridge`` (KDE rows), ``Bar`` (mean ± CI), or ``Strip`` (jittered points).

    Returns
    -------
    GlobalAttributionResult
        Contains the figure and summary statistics.  Check ``.warning`` before
        rendering the figure — it is ``None`` when attributions were found.
    """
    combined_explanations = compute_and_cache_masking(
        models=models,
        experiment_path=experiment_path,
        dataset=dataset,
        explain_mols=explain_mols,
        problem_type=problem_type,
        fragmentation_approach=fragmentation_approach,
        target_class=target_class,
    )

    display_data = build_display_data(combined_explanations, models, explain_mols)

    fk = _frag_key(fragmentation_approach)
    ck = _class_key(target_class)

    all_frags = {
        frag
        for model_data in display_data.values()
        for num_data in model_data.values()
        for mol_data in num_data.values()
        for mon_data in mol_data.get(_ALG_KEY, {}).get(ck, {}).values()
        for frag in mon_data.get(fk, {})
    }
    n_mols = sum(
        len(num_data) for model_data in display_data.values() for num_data in model_data.values()
    )
    n_models = len(models)
    n_frags_total = len(all_frags)
    n_shown = (
        min(top_n * 2, n_frags_total) if top_n and n_frags_total > top_n * 2 else n_frags_total
    )

    attribution_dict = fragment_attributions_to_distribution(
        node_masks=display_data,
        model_log_names=list(models.keys()),
        target_class=target_class,
        fragmentation_method=fragmentation_approach,
        normalisation_type=normalisation_type,
    )

    base_result = dict(
        attribution_dict=attribution_dict,
        n_mols=n_mols,
        n_models=n_models,
        n_frags_total=n_frags_total,
        n_shown=n_shown,
        target_class=target_class,
        normalisation_type=normalisation_type,
    )

    if not attribution_dict:
        return GlobalAttributionResult(
            figure=None,
            warning="No fragment attributions found for the selected molecules and models.",
            **base_result,
        )

    shared_kwargs = dict(
        attribution_dict=attribution_dict, neg_color=neg_color, pos_color=pos_color, top_n=top_n
    )

    if plot_type == AttributionPlotType.Bar:
        fig = plot_attribution_bar(**shared_kwargs)
    elif plot_type == AttributionPlotType.Strip:
        fig = plot_attribution_strip(**shared_kwargs)
    else:  # Ridge (default)
        fig = plot_attribution_distribution(**shared_kwargs)

    return GlobalAttributionResult(figure=fig, **base_result)


# ---------------------------------------------------------------------------
# High-level: local (per-molecule) attribution
# ---------------------------------------------------------------------------


def compute_local_attribution(
    models: dict,
    experiment_path: Path,
    dataset,
    explain_mols: list,
    problem_type: ProblemType,
    neg_color: str = "#40bcde",
    pos_color: str = "#e64747",
    normalisation_type: ImportanceNormalisationMethod = ImportanceNormalisationMethod.PerModel,
    fragmentation_approach=FragmentationMethod.BRICS,
    target_class: int | None = None,
    mol_names: dict | None = None,
    predictions: dict | None = None,
    class_labels: dict | None = None,
) -> list[MolAttributionResult]:
    """
    Compute per-molecule attribution tables and atom heatmap figures.

    Parameters
    ----------
    models:
        ``{"{model_name}_{number}": model}`` ensemble.
    experiment_path:
        Root path of the PolyNet experiment directory.
    dataset:
        The PyG dataset to draw molecules from.
    explain_mols:
        Molecule IDs to display — each gets its own result entry.
    problem_type:
        Regression or classification.
    neg_color:
        Hex colour for the negative-attribution end of the colormap.
    pos_color:
        Hex colour for the positive-attribution end.
    normalisation_type:
        Normalisation strategy (Local, Global, PerModel, NoNormalisation).
    fragmentation_approach:
        Fragmentation strategy (default: BRICS).
    target_class:
        For classification, the class index to attribute.
    mol_names:
        ``{mol_idx: monomer_name}`` — optional display names for monomers.
    predictions:
        ``{mol_idx: {ResultColumn.LABEL: ..., ResultColumn.PREDICTED: ...}}``.
    class_labels:
        ``{class_int: class_name}`` — currently reserved for future use.

    Returns
    -------
    list[MolAttributionResult]
        One entry per molecule in ``explain_mols``, in dataset order.
        Check ``.warning`` before rendering — it is ``None`` when a fragment
        match was found.
    """
    mol_names = mol_names or {}
    predictions = predictions or {}
    class_labels = class_labels or {}

    combined_explanations = compute_and_cache_masking(
        models=models,
        experiment_path=experiment_path,
        dataset=dataset,
        explain_mols=explain_mols,
        problem_type=problem_type,
        fragmentation_approach=fragmentation_approach,
        target_class=target_class,
    )

    display_data = build_display_data(combined_explanations, models, explain_mols)

    fk = _frag_key(fragmentation_approach)
    ck = _class_key(target_class)

    merged = merge_fragment_attributions(display_data, list(models.keys()))

    # Global divisor derived from the averaged merged data
    if normalisation_type == ImportanceNormalisationMethod.Global:
        global_divisor = (
            max(
                (
                    abs(s)
                    for mol_data in merged.values()
                    for mon_data in mol_data.get(_ALG_KEY, {}).get(ck, {}).values()
                    for scores in mon_data.get(fk, {}).values()
                    for s in scores
                ),
                default=1.0,
            )
            or 1.0
        )
    else:
        global_divisor = None

    mols_filtered = filter_dataset_by_ids(dataset, explain_mols)
    cmap = get_cmap(neg_color=neg_color, pos_color=pos_color)

    results: list[MolAttributionResult] = []

    for mol in mols_filtered:
        info_msg = (
            f"Fragment attributions for `{mol.idx}` — Chemistry Masking ({fk})"
            + (f", class `{target_class}`" if target_class is not None else "")
            + f" | normalisation: `{normalisation_type}`"
        )
        true_label = str(predictions.get(mol.idx, {}).get(ResultColumn.LABEL, "N/A"))
        predicted_label = str(predictions.get(mol.idx, {}).get(ResultColumn.PREDICTED, "N/A"))

        monomer_dict = merged.get(mol.idx, {}).get(_ALG_KEY, {}).get(ck, {})
        if not monomer_dict:
            results.append(
                MolAttributionResult(
                    mol_idx=mol.idx,
                    info_msg=info_msg,
                    true_label=true_label,
                    predicted_label=predicted_label,
                    attribution_df=pd.DataFrame(),
                    mol_figure=None,
                    warning="No fragments matched for this molecule.",
                )
            )
            continue

        mol_raw_scores = [
            s
            for mon_data in monomer_dict.values()
            for scores in mon_data.get(fk, {}).values()
            for s in scores
        ]

        if normalisation_type in (
            ImportanceNormalisationMethod.Local,
            ImportanceNormalisationMethod.PerModel,
        ):
            mol_divisor = max((abs(s) for s in mol_raw_scores), default=1.0) or 1.0
        elif normalisation_type == ImportanceNormalisationMethod.Global:
            mol_divisor = global_divisor
        else:
            mol_divisor = None

        attr_col = "Attribution (Y − Y_masked)"
        norm_col = "Normalised Attribution" if mol_divisor is not None else attr_col
        rows = [
            {
                "Monomer (SMILES)": monomer_smiles,
                "Fragment (SMILES)": frag_smiles,
                "Occurrence": occ + 1,
                attr_col: score,
            }
            for monomer_smiles, fk_dict in monomer_dict.items()
            for frag_smiles, scores in fk_dict.get(fk, {}).items()
            for occ, score in enumerate(scores)
        ]
        df = pd.DataFrame(rows).sort_values(attr_col, ascending=False)
        if mol_divisor is not None:
            df[norm_col] = df[attr_col] / mol_divisor

        weights_list = fragment_attributions_to_atom_weights(
            mol=mol,
            monomer_dict=monomer_dict,
            frag_key=fk,
            fragmentation_approach=fragmentation_approach,
        )

        if mol_divisor is not None:
            weights_list = [[w / mol_divisor for w in wlist] for wlist in weights_list]
            cmap_min, cmap_max = -1.0, 1.0
        else:
            cmap_min, cmap_max = None, None

        names = mol_names.get(mol.idx, None)
        fig = plot_mols_with_weights(
            smiles_list=mol.mols,
            weights_list=weights_list,
            colormap=cmap,
            legend=names,
            min_weight=cmap_min,
            max_weight=cmap_max,
        )

        results.append(
            MolAttributionResult(
                mol_idx=mol.idx,
                info_msg=info_msg,
                true_label=true_label,
                predicted_label=predicted_label,
                attribution_df=df,
                mol_figure=fig,
            )
        )

    return results
