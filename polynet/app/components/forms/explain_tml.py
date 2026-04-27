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
from polynet.config.enums import AttributionPlotType, ImportanceNormalisationMethod, ProblemType
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
) -> None:
    st.markdown(
        "**Which molecular features drive predictions across the population?**  \n"
        "Select a set of samples and run to see a distribution of SHAP attributions "
        "across all selected models. Each data point represents one real model prediction, "
        "preserving the full ensemble spread."
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
            options=[AttributionPlotType.Ridge, AttributionPlotType.Bar, AttributionPlotType.Strip],
            format_func=lambda x: {
                AttributionPlotType.Ridge: "Ridge (full distribution)",
                AttributionPlotType.Bar: "Bar (mean ± 95 % CI)",
                AttributionPlotType.Strip: "Strip (individual scores + mean)",
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
        )


# ---------------------------------------------------------------------------
# Local explanation tab
# ---------------------------------------------------------------------------


def _tml_local_tab(
    shared: dict,
    models: dict,
    descriptor_dfs: dict[str, pd.DataFrame],
    experiment_path: Path,
    preds: pd.DataFrame,
    data_options: DataConfig,
) -> None:
    st.markdown(
        "**How do features influence a specific sample's prediction?**  \n"
        "Select one or more samples to see per-instance SHAP plots and attribution tables. "
        "SHAP values are averaged across the selected ensemble models."
    )

    local_samples = st.multiselect(
        "Select samples to explain",
        options=sorted(preds.index.tolist()),
        key=TMLExplainStateKeys.LocalTMLIDSelector,
        default=None,
    )

    if not local_samples:
        st.info("Select at least one sample above to generate local SHAP explanations.")
        return

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

    # Build predictions dict for true / predicted label display
    true_col = get_true_label_column_name(data_options.target_variable_name)
    pred_col = None
    for key in models:
        # key format: "{algo}-{descriptor}_{iteration}" → ml_model = "{algo}-{descriptor}"
        ml_model = key.rsplit("_", 1)[0]
        candidate = get_predicted_label_column_name(data_options.target_variable_name, ml_model)
        if candidate in preds.columns:
            pred_col = candidate
            break

    preds_dedup = preds[~preds.index.duplicated(keep="first")]
    class_names = data_options.class_names
    preds_dict: dict = {}
    for sid in local_samples:
        preds_dict[sid] = {}
        try:
            if true_col in preds_dedup.columns and sid in preds_dedup.index:
                if data_options.problem_type == ProblemType.Regression:
                    preds_dict[sid]["true"] = str(preds_dedup.loc[sid, true_col])
                else:
                    # Convert to int → str key to mirror the GNN class_names lookup
                    true_str = str(int(preds_dedup.loc[sid, true_col]))
                    preds_dict[sid]["true"] = class_names[true_str] if class_names else true_str
            if pred_col and sid in preds_dedup.index:
                if data_options.problem_type == ProblemType.Regression:
                    preds_dict[sid]["predicted"] = str(preds_dedup.loc[sid, pred_col])
                else:
                    pred_str = str(int(preds_dedup.loc[sid, pred_col]))
                    preds_dict[sid]["predicted"] = class_names[pred_str] if class_names else pred_str
        except (KeyError, TypeError, ValueError):
            pass

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
            predictions=preds_dict,
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
        )

    with local_tab:
        _tml_local_tab(
            shared=shared,
            models=filtered_models,
            descriptor_dfs=filtered_dfs,
            experiment_path=experiment_path,
            preds=preds,
            data_options=data_options,
        )
