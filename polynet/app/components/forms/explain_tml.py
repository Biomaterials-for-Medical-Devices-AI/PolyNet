"""
polynet.app.components.forms.explain_tml
==========================================
Streamlit UI form for TML SHAP explainability.

Mirrors the structure of ``explain_model.py`` (GNN) but works on descriptor
DataFrames instead of molecular graphs.  All computation is delegated to
``polynet.app.services.explain_tml``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from polynet.app.components.forms.explain_model import explain_mols_widget
from polynet.app.options.state_keys import TMLExplainStateKeys
from polynet.app.services.explain_tml import explain_tml_global, explain_tml_local
from polynet.config.column_names import get_predicted_label_column_name, get_true_label_column_name
from polynet.config.constants import ResultColumn
from polynet.config.display_names import prettify_label
from polynet.config.enums import (
    ExplanationAggregation,
    ImportanceNormalisationMethod,
    IteratorType,
    ProblemType,
    ShapGlobalPlotType,
)
from polynet.config.schemas import DataConfig

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------


def _tml_shared_params_section(
    tml_model_log_names: list[str],
    descriptor_dfs: dict[str, pd.DataFrame],
    data_options: DataConfig,
) -> dict | None:
    """
    Render parameters shared by both explanation tabs.

    Returns a params dict, or ``None`` + stops rendering if nothing is
    selected.
    """
    all_model_types = sorted({key.split("-", 1)[0] for key in tml_model_log_names})
    all_descriptors = sorted(
        {key.split("-", 1)[1].rsplit("_", 1)[0] for key in tml_model_log_names}
    )
    all_iterations = sorted({int(key.rsplit("_", 1)[1]) for key in tml_model_log_names})

    select_all = st.toggle(
        "Select all TML models", key=TMLExplainStateKeys.SelectAllTMLModels, value=False
    )
    selected_model_types = st.multiselect(
        "Select TML model types to explain",
        options=all_model_types,
        default=all_model_types if select_all else None,
        key=TMLExplainStateKeys.TMLModelSelector,
    )

    selected_reprs = st.multiselect(
        "Select descriptor representations to explain",
        options=all_descriptors,
        default=all_descriptors,
        key=TMLExplainStateKeys.TMLRepresentationSelector,
    )

    selected_iterations = st.multiselect(
        "Select bootstrap iterations",
        options=all_iterations,
        default=all_iterations,
        key=TMLExplainStateKeys.TMLBootstrapSelector,
        help="Each iteration is one train/test split. Average SHAP values across iterations for a more robust estimate.",
    )

    if not selected_model_types or not selected_reprs or not selected_iterations:
        st.info("Select at least one model type, one representation, and one bootstrap iteration.")
        return None

    # Target class (classification only)
    target_class = None
    if data_options.problem_type == ProblemType.Classification:
        if data_options.class_names:
            options = list(data_options.class_names.values())
        else:
            options = list(range(data_options.num_classes))

        selected_class_label = st.selectbox(
            "Target class to explain",
            options=options,
            index=0,
            key=TMLExplainStateKeys.TargetClassTML,
            help="The class index whose SHAP values are computed.",
        )
        if data_options.class_names:
            class_num = next(
                (k for k, v in data_options.class_names.items() if v == selected_class_label), None
            )
            st.info(f"`{selected_class_label}` corresponds to class index `{class_num}`.")
            target_class = int(class_num)

    with st.expander("Advanced settings", expanded=False):
        normalisation_type = st.radio(
            "Normalisation",
            options=[
                ImportanceNormalisationMethod.Local,
                ImportanceNormalisationMethod.PerModel,
                ImportanceNormalisationMethod.Global,
                ImportanceNormalisationMethod.NoNormalisation,
            ],
            key=TMLExplainStateKeys.NormalisationTML,
            index=1,
            horizontal=True,
            help=(
                "**Local** — each (model × sample) unit scaled to [−1, 1] independently.  \n"
                "**PerModel** — each model scaled by its own maximum SHAP value.  \n"
                "**Global** — single divisor across all models and samples.  \n"
                "**None** — raw SHAP values."
            ),
        )

        cols = st.columns(2)
        with cols[0]:
            neg_color = st.color_picker(
                "Negative attribution colour", key=TMLExplainStateKeys.NegColorTML, value="#40bcde"
            )
        with cols[1]:
            pos_color = st.color_picker(
                "Positive attribution colour", key=TMLExplainStateKeys.PosColorTML, value="#e64747"
            )

    return {
        "selected_model_types": selected_model_types,
        "selected_reprs": selected_reprs,
        "selected_iterations": set(selected_iterations),
        "normalisation_type": normalisation_type,
        "target_class": target_class,
        "neg_color": neg_color,
        "pos_color": pos_color,
    }


# ---------------------------------------------------------------------------
# Global explanation tab
# ---------------------------------------------------------------------------


def _tml_global_tab(
    shared: dict,
    models: dict,
    descriptor_dfs: dict[str, pd.DataFrame],
    experiment_path: Path,
    preds: pd.DataFrame,
    data_options: DataConfig,
    cache_root: Path | None = None,
) -> None:
    st.markdown(
        "**Which molecular features drive predictions across the population?**  \n"
        "Select a set of samples and run to see a native SHAP summary "
        "(beeswarm / bar / violin). SHAP values are averaged across the selected "
        "ensemble models, giving one attribution per sample per feature."
    )

    explain_samples = explain_mols_widget(
        data=preds,
        SetStateKey=TMLExplainStateKeys.GlobalTMLExplainSet,
        ManuallySelectStateKey=TMLExplainStateKeys.GlobalTMLManualSelector,
        MolsStateKey=TMLExplainStateKeys.GlobalTMLIDSelector,
    )

    cols = st.columns(2)
    with cols[0]:
        plot_type = st.radio(
            "Plot type",
            options=[
                ShapGlobalPlotType.Beeswarm,
                ShapGlobalPlotType.Bar,
                ShapGlobalPlotType.Violin,
            ],
            format_func=lambda x: {
                ShapGlobalPlotType.Beeswarm: "Beeswarm (per-sample, coloured by feature value)",
                ShapGlobalPlotType.Bar: "Bar (mean |SHAP|)",
                ShapGlobalPlotType.Violin: "Violin (per-feature distribution)",
            }[x],
            key=TMLExplainStateKeys.GlobalTMLPlotType,
            horizontal=False,
        )
    with cols[1]:
        top_n = st.number_input(
            "Features to show (top N)",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            key=TMLExplainStateKeys.GlobalTMLTopN,
            help="Show the N features with the highest mean |SHAP| value.",
        )

    if st.button("Run Global SHAP Explanation") or st.toggle(
        "Keep running automatically", key=TMLExplainStateKeys.GlobalTMLKeepRunning
    ):
        explain_tml_global(
            models=models,
            descriptor_dfs=descriptor_dfs,
            experiment_path=experiment_path,
            explain_sample_ids=explain_samples,
            problem_type=data_options.problem_type,
            neg_color=shared["neg_color"],
            pos_color=shared["pos_color"],
            normalisation_type=shared["normalisation_type"],
            target_class=shared["target_class"],
            top_n=int(top_n),
            plot_type=plot_type,
            cache_root=cache_root,
            target_col=data_options.target_variable_col,
        )


# ---------------------------------------------------------------------------
# Local explanation tab
# ---------------------------------------------------------------------------


def _build_prediction_breakdowns(
    preds: pd.DataFrame,
    model_log_names: list[str],
    sample_ids: list,
    data_options: DataConfig,
    iterator_col: str | None,
    selected_iterations,
) -> dict[str, dict[str, dict]]:
    """
    Build a per (descriptor, sample) prediction breakdown for the local panels.

    Predictions hold one row per (sample × bootstrap iteration), and each
    selected model is a specific ``(algorithm, representation, bootstrap)``. For
    every selected model × bootstrap this records the predicted value and the
    set (train/val/test) the sample belonged to in *that* bootstrap, plus the
    sample's true label.

    Returns
    -------
    dict
        ``{descriptor: {str(sample_id): {"true": str, "rows": [
            {"Model": str, "Bootstrap": int, "Set": str, "Predicted": str}, ...
        ]}}}``
    """
    target = data_options.target_variable_name
    true_col = get_true_label_column_name(target)
    class_names = data_options.class_names
    is_regression = data_options.problem_type == ProblemType.Regression

    def _fmt(value) -> str:
        if value is None or pd.isna(value):
            return "N/A"
        if is_regression:
            try:
                return f"{float(value):.4g}"
            except (TypeError, ValueError):
                return str(value)
        key = str(int(value))
        return class_names[key] if class_names else key

    # (sample_id, iteration) -> row, for O(1) lookup.
    row_lookup: dict = {}
    if iterator_col and iterator_col in preds.columns:
        tmp = preds.copy()
        tmp["_sid"] = tmp.index
        for (sid_k, it_k), sub in tmp.groupby(["_sid", iterator_col]):
            row_lookup[(sid_k, int(it_k))] = sub.iloc[0]

    breakdowns: dict[str, dict[str, dict]] = {}
    for key in model_log_names:
        # key format: "{algo}-{descriptor}_{iteration}"
        algo_descr, it_str = key.rsplit("_", 1)
        descriptor = algo_descr.split("-", 1)[1]
        iteration = int(it_str)
        if selected_iterations and iteration not in selected_iterations:
            continue

        pred_col = get_predicted_label_column_name(target, algo_descr)
        model_label = prettify_label(algo_descr)
        desc_map = breakdowns.setdefault(descriptor, {})

        for sid in sample_ids:
            entry = desc_map.setdefault(str(sid), {"true": "N/A", "rows": []})
            row = row_lookup.get((sid, iteration))
            # Always emit a row per selected model × bootstrap. When the sample
            # was not part of this bootstrap's data there is no stored prediction,
            # so Set/Predicted show "N/A" rather than the row being dropped.
            if row is not None and true_col in row.index:
                entry["true"] = _fmt(row[true_col])
            entry["rows"].append(
                {
                    "Model": model_label,
                    "Bootstrap": iteration,
                    "Set": (
                        row[ResultColumn.SET]
                        if row is not None and ResultColumn.SET in row.index
                        else "N/A"
                    ),
                    "Predicted": (
                        _fmt(row[pred_col]) if row is not None and pred_col in row.index else "N/A"
                    ),
                }
            )

    # Stable ordering for display.
    for desc_map in breakdowns.values():
        for entry in desc_map.values():
            entry["rows"].sort(key=lambda r: (r["Model"], r["Bootstrap"]))

    return breakdowns


def _tml_local_tab(
    shared: dict,
    models: dict,
    descriptor_dfs: dict[str, pd.DataFrame],
    experiment_path: Path,
    preds: pd.DataFrame,
    data_options: DataConfig,
    cache_root: Path | None = None,
) -> None:
    st.markdown(
        "**How do features influence a specific sample's prediction?**  \n"
        "Select one or more samples to see per-instance SHAP plots and attribution tables. "
        "SHAP values are averaged across the selected ensemble models."
    )

    # Predictions hold one row per (sample × bootstrap iteration), so the index
    # repeats. Restrict to the selected iterations and drop duplicate sample ids
    # so each sample is offered exactly once (mirrors the GNN local selector).
    iterator_cols = [c for c in preds.columns if c in {it.value for it in IteratorType}]
    selected_iterations = shared.get("selected_iterations")
    preds_unique = preds
    if iterator_cols and selected_iterations:
        preds_unique = preds_unique[preds_unique[iterator_cols[0]].isin(selected_iterations)]
    preds_unique = preds_unique[~preds_unique.index.duplicated(keep="first")]

    local_samples = st.multiselect(
        "Select samples to explain",
        options=sorted(preds_unique.index.tolist()),
        key=TMLExplainStateKeys.LocalTMLIDSelector,
        default=None,
    )

    if not local_samples:
        st.info("Select at least one sample above to generate local SHAP explanations.")
        return

    cols = st.columns(2)
    with cols[0]:
        local_plot_type = st.radio(
            "Local plot style",
            options=["waterfall", "force", "bar"],
            key=TMLExplainStateKeys.LocalTMLPlotType,
            horizontal=True,
            help=(
                "**Waterfall** — cumulative contribution chart.  \n"
                "**Force** — stacked horizontal force plot.  \n"
                "**Bar** — simple ranked horizontal bar chart."
            ),
        )
    with cols[1]:
        local_top_n = st.number_input(
            "Features to show (top N)",
            min_value=1,
            max_value=100,
            value=15,
            step=1,
            key=TMLExplainStateKeys.LocalTMLTopN,
            help="Maximum number of features (by |SHAP|) to display. "
            "Applies to the waterfall and bar plots; the force plot shows all.",
        )

    aggregation = st.radio(
        "Models display",
        options=[ExplanationAggregation.Average, ExplanationAggregation.Separate],
        format_func=lambda x: {
            ExplanationAggregation.Average: "Average across models (one plot per sample)",
            ExplanationAggregation.Separate: "Show each model separately (one plot per model)",
        }[x],
        key=TMLExplainStateKeys.LocalTMLAggregation,
        horizontal=True,
        help="How to combine explanations from the selected models for the same sample.",
    )

    # Per (descriptor, sample) breakdown: true label + each selected
    # model × bootstrap's predicted value and set membership in that bootstrap.
    prediction_breakdowns = _build_prediction_breakdowns(
        preds=preds,
        model_log_names=list(models.keys()),
        sample_ids=local_samples,
        data_options=data_options,
        iterator_col=iterator_cols[0] if iterator_cols else None,
        selected_iterations=selected_iterations,
    )

    if st.button("Run Local SHAP Explanation") or st.toggle(
        "Keep running automatically", key=TMLExplainStateKeys.LocalTMLKeepRunning
    ):
        explain_tml_local(
            models=models,
            descriptor_dfs=descriptor_dfs,
            experiment_path=experiment_path,
            explain_sample_ids=local_samples,
            problem_type=data_options.problem_type,
            neg_color=shared["neg_color"],
            pos_color=shared["pos_color"],
            normalisation_type=shared["normalisation_type"],
            target_class=shared["target_class"],
            local_plot_type=local_plot_type,
            max_display=int(local_top_n),
            prediction_breakdowns=prediction_breakdowns,
            cache_root=cache_root,
            target_col=data_options.target_variable_col,
            aggregation=aggregation,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def explain_tml_form(
    experiment_path: Path,
    tml_models: dict,
    descriptor_dfs: dict[str, pd.DataFrame],
    data_options: DataConfig,
    preds: pd.DataFrame,
    cache_root: Path | None = None,
) -> None:
    """
    Render the full TML SHAP explainability form with shared params + two tabs.

    Parameters
    ----------
    experiment_path:
        Experiment root directory (for SHAP cache and saving outputs).
    tml_models:
        Trained TML models keyed by log name ``"{ModelType}-{descriptor}_{iter}"``.
    descriptor_dfs:
        Descriptor DataFrames keyed by descriptor name (e.g. ``"morgan"``).
    data_options:
        Data configuration (problem type, class names, etc.).
    preds:
        Predictions DataFrame with ``SET`` column and sample IDs as index.
    """
    model_log_names = list(tml_models.keys())

    shared = _tml_shared_params_section(
        tml_model_log_names=model_log_names,
        descriptor_dfs=descriptor_dfs,
        data_options=data_options,
    )
    if shared is None:
        return

    # Filter models and descriptor_dfs to the selected subset
    selected_types = set(shared["selected_model_types"])
    selected_reprs = set(shared["selected_reprs"])

    selected_iterations = shared["selected_iterations"]
    filtered_models = {
        key: model
        for key, model in tml_models.items()
        if key.split("-", 1)[0] in selected_types
        and key.split("-", 1)[1].rsplit("_", 1)[0] in selected_reprs
        and int(key.rsplit("_", 1)[1]) in selected_iterations
    }
    filtered_dfs = {k: v for k, v in descriptor_dfs.items() if k in selected_reprs}

    if not filtered_models:
        st.warning("No models match the selected filters.")
        return

    global_tab, local_tab = st.tabs(["Global Explanation", "Local Explanation"])

    with global_tab:
        _tml_global_tab(
            shared=shared,
            models=filtered_models,
            descriptor_dfs=filtered_dfs,
            experiment_path=experiment_path,
            preds=preds,
            data_options=data_options,
            cache_root=cache_root,
        )

    with local_tab:
        _tml_local_tab(
            shared=shared,
            models=filtered_models,
            descriptor_dfs=filtered_dfs,
            experiment_path=experiment_path,
            preds=preds,
            data_options=data_options,
            cache_root=cache_root,
        )
