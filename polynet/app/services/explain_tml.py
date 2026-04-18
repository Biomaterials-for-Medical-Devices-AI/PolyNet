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

from polynet.config.enums import AttributionPlotType, ImportanceNormalisationMethod, ProblemType
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
    plot_type: AttributionPlotType = AttributionPlotType.Ridge,
) -> None:
    """Compute and render the global SHAP attribution distribution (one section per descriptor)."""
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
) -> None:
    """Compute and render per-instance SHAP panels (table + plot)."""
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
    )

    for descriptor, instance_results in results.items():
        st.markdown(f"**{descriptor}**")
        for inst in instance_results:
            container = st.container(
                border=True, key=f"tml_local_{descriptor}_{inst.sample_idx}_container"
            )
            container.info(inst.info_msg)
            container.write(f"True label: `{inst.true_label}`")
            container.write(f"Predicted label: `{inst.predicted_label}`")

            if inst.warning:
                container.warning(inst.warning)
                continue

            container.dataframe(inst.attribution_df, use_container_width=True)
            if inst.figure is not None:
                container.pyplot(inst.figure, use_container_width=True)
