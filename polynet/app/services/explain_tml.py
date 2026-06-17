"""
polynet.app.services.explain_tml
==================================
Thin Streamlit rendering layer for TML SHAP explainability.

Calls the pure-computation functions from ``polynet.explainability.shap_explain``
and renders results with ``st.pyplot`` / ``st.dataframe``. No computation lives here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from polynet.config.enums import (
    ExplanationAggregation,
    ImportanceNormalisationMethod,
    ProblemType,
    ShapGlobalPlotType,
)
from polynet.explainability.shap_explain import (
    GlobalAttributionResult,
    InstanceAttributionResult,
    compute_global_shap_attribution,
    compute_local_shap_attribution,
)


def explain_tml_global(
    models: dict,
    descriptor_dfs: dict[str, pd.DataFrame],
    experiment_path: Path,
    explain_sample_ids: list[str],
    problem_type: ProblemType,
    neg_color: str = "#40bcde",
    pos_color: str = "#e64747",
    normalisation_type: ImportanceNormalisationMethod = ImportanceNormalisationMethod.PerModel,
    target_class: int | None = None,
    top_n: int | None = 10,
    plot_type: ShapGlobalPlotType = ShapGlobalPlotType.Beeswarm,
    cache_root: Path | None = None,
    target_col: str | None = None,
) -> None:
    """Compute and render the global SHAP summary (one section per descriptor)."""
    results: dict[str, GlobalAttributionResult] = compute_global_shap_attribution(
        models=models,
        descriptor_dfs=descriptor_dfs,
        experiment_path=experiment_path,
        problem_type=problem_type,
        explain_sample_ids=explain_sample_ids,
        neg_color=neg_color,
        pos_color=pos_color,
        normalisation_type=normalisation_type,
        target_class=target_class,
        top_n=top_n,
        plot_type=plot_type,
        cache_root=cache_root,
        target_col=target_col,
    )

    for descriptor, result in results.items():
        st.markdown(f"**{descriptor}**")
        st.info(
            f"Distribution over **{result.n_mols}** sample(s) × **{result.n_models}** model(s) — "
            f"**{result.n_frags_total}** feature(s) found, showing top **{result.n_shown}**."
            + (f" | class `{result.target_class}`" if result.target_class is not None else "")
            + f" | normalisation: `{result.normalisation_type}`"
        )
        if result.warning:
            st.warning(result.warning)
        else:
            st.pyplot(result.figure, use_container_width=True)


def explain_tml_local(
    models: dict,
    descriptor_dfs: dict[str, pd.DataFrame],
    experiment_path: Path,
    explain_sample_ids: list[str],
    problem_type: ProblemType,
    neg_color: str = "#40bcde",
    pos_color: str = "#e64747",
    normalisation_type: ImportanceNormalisationMethod = ImportanceNormalisationMethod.PerModel,
    target_class: int | None = None,
    local_plot_type: str = "waterfall",
    max_display: int = 15,
    predictions: dict | None = None,
    prediction_breakdowns: dict | None = None,
    cache_root: Path | None = None,
    target_col: str | None = None,
    aggregation: ExplanationAggregation = ExplanationAggregation.Average,
) -> None:
    """Compute and render per-instance SHAP panels (table + plot).

    ``prediction_breakdowns`` (optional) holds, per descriptor and sample, the
    true label and one row per selected model × bootstrap with that bootstrap's
    predicted value and set membership. When provided it replaces the single
    true/predicted label lines.

    ``aggregation`` controls whether the selected models are merged into one
    plot per sample (``Average``) or shown as one plot per model × sample
    (``Separate``).
    """
    results: dict[str, list[InstanceAttributionResult]] = compute_local_shap_attribution(
        models=models,
        descriptor_dfs=descriptor_dfs,
        experiment_path=experiment_path,
        problem_type=problem_type,
        explain_sample_ids=explain_sample_ids,
        neg_color=neg_color,
        pos_color=pos_color,
        normalisation_type=normalisation_type,
        target_class=target_class,
        local_plot_type=local_plot_type,
        max_display=max_display,
        predictions=predictions,
        cache_root=cache_root,
        target_col=target_col,
        aggregation=aggregation,
    )

    for descriptor, instance_results in results.items():
        st.markdown(f"**{descriptor}**")

        # Group by sample (preserving order). In Average mode each sample has one
        # result (model_label is None); in Separate mode it has one per model.
        by_sample: dict[str, list[InstanceAttributionResult]] = {}
        for inst in instance_results:
            by_sample.setdefault(str(inst.sample_idx), []).append(inst)

        for sid, insts in by_sample.items():
            container = st.container(border=True, key=f"tml_local_{descriptor}_{sid}_container")
            container.info(insts[0].info_msg)

            breakdown = (prediction_breakdowns or {}).get(descriptor, {}).get(sid)
            if breakdown is not None:
                container.write(f"True label: `{breakdown['true']}`")
                if breakdown["rows"]:
                    container.dataframe(
                        pd.DataFrame(breakdown["rows"]),
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                container.write(f"True label: `{insts[0].true_label}`")
                container.write(f"Predicted label: `{insts[0].predicted_label}`")

            for inst in insts:
                if inst.model_label:
                    container.markdown(f"**{inst.model_label}**")
                if inst.warning:
                    container.warning(inst.warning)
                    continue
                container.dataframe(inst.attribution_df, use_container_width=True)
                if inst.figure is not None:
                    container.pyplot(inst.figure, use_container_width=True)
