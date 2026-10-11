"""Page 6 controls for rendering completed explanations, without inference."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from polynet.app.services.cache import file_version
from polynet.app.services.explain_model import display_gnn_global, display_gnn_local
from polynet.app.services.explain_source import list_external_datasets, resolve_explain_source
from polynet.app.services.explain_tml import display_tml_global, display_tml_local
from polynet.app.services.explanation_results import (
    analysis_label,
    analysis_summary,
    attribution_coverage,
    list_saved_analyses,
    load_analysis,
    local_prediction_breakdowns,
)
from polynet.config.enums import (
    AttributionPlotType,
    ExplanationAggregation,
    ImportanceNormalisationMethod,
    ShapGlobalPlotType,
)
from polynet.explainability.explain import (
    render_cached_global_attribution,
    render_cached_local_attribution,
)
from polynet.explainability.shap_explain import (
    render_cached_global_shap_attribution,
    render_cached_local_shap_attribution,
)


def explanation_plots_form(experiment_path: Path, data_options) -> None:
    """One saved analysis, selectable population/molecule views, no inference."""
    st.subheader("Feature importance plots")
    sources = {"Experiment data": experiment_path}
    for name in list_external_datasets(experiment_path):
        sources[f"External dataset: {name}"] = resolve_explain_source(
            experiment_path, name
        ).cache_root
    available = {label: root for label, root in sources.items() if list_saved_analyses(root)}
    if not available:
        st.info(
            "No feature importance analysis found. Run an explanation on page 5 — Explain Models first."
        )
        return
    if not st.checkbox("Customise feature importance plots", key="analyse_explanations_enabled"):
        return
    label = (
        st.selectbox("Explanation data source", list(available), key="analyse_explanation_source")
        if len(available) > 1
        else next(iter(available))
    )
    root = available[label]
    paths = list_saved_analyses(root)
    try:
        labels = {path: analysis_label(analysis_summary(path)) for path in paths}
        # Rare equal labels get a simple index; internal hashes/dates stay hidden.
        duplicate_labels = {
            value for value in labels.values() if list(labels.values()).count(value) > 1
        }
        for i, path in enumerate(paths, 1):
            if labels[path] in duplicate_labels:
                labels[path] += f" (analysis {i})"
        path = (
            st.selectbox(
                "Feature importance analysis",
                paths,
                format_func=labels.get,
                key="analyse_explanation_snapshot",
            )
            if len(paths) > 1
            else paths[0]
        )
        if len(paths) == 1:
            st.caption(labels[path])
        identity = repr(file_version(path))
        loaded_key = f"loaded_explanation_{experiment_path}"
        loaded = st.session_state.get(loaded_key)
        if loaded is None or loaded[0] != identity:
            loaded = (identity, load_analysis(path))
            st.session_state[loaded_key] = loaded
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Could not load this analysis: {exc}")
        return
    analysis = loaded[1]
    args = analysis["arguments"].copy()
    family = analysis["family"]
    coverage = attribution_coverage(analysis)
    if not coverage:
        st.info("This analysis has no saved attribution values. Run it again on page 5.")
        return
    prefix = "explanation_plot_" + hashlib.sha256(identity.encode()).hexdigest()[:12]
    view = st.radio(
        "Plot view", ["Population", "Individual molecules"], horizontal=True, key=f"{prefix}_view"
    )
    scope = "global" if view == "Population" else "local"
    selected = list(coverage)
    if scope == "local":
        selected = st.multiselect(
            "Molecules to display",
            sorted(coverage),
            default=[sorted(coverage)[0]],
            key=f"{prefix}_molecules",
        )
        if not selected:
            st.info("Select at least one explained molecule.")
            return
        available_names = sorted(set().union(*(coverage[sid] for sid in selected)))
        args["model_names"] = st.multiselect(
            "Cached models / bootstraps to show",
            available_names,
            default=available_names,
            key=f"{prefix}_models",
        )
        if not args["model_names"]:
            st.info("Select at least one cached model/bootstrap combination.")
            return
        for sid in selected:
            names = sorted(coverage[sid] & set(args["model_names"]))
            st.caption(
                f"{sid}: showing cached explanations from "
                + (", ".join(names) or "none of the selected combinations")
            )
            missing = set(args["model_names"]) - coverage[sid]
            if missing:
                st.info(
                    f"{sid} was not explained by {', '.join(sorted(missing))}. This plot uses the saved combinations listed above. Run page 5 for additional combinations."
                )
        if family == "GNN":
            stored_ids = {str(mol.idx): mol.idx for mol in args["dataset"]}
            args["explain_mols"] = [stored_ids[sid] for sid in selected]
            # Keep the full raw population for the PerModel reference maximum.
            args["dataset"] = [mol for mol in args["dataset"] if str(mol.idx) in selected]
        else:
            args["explain_sample_ids"] = selected
            st.caption(
                "PerModel uses each model/bootstrap's largest absolute attribution across the whole saved analysis. Normalised plots show scaled model output."
            )
            if args["problem_type"] == "classification":
                st.caption(
                    "SHAP output scales can differ by model (probability or margin/log-odds). Use Separate model display to interpret individual endpoints."
                )
    st.caption(
        f"{len(coverage)} explained molecules · {len(args['model_names'])} model/bootstrap combinations"
    )
    _plot_controls(args, family, scope, prefix + "_" + scope)
    renderer, display = {
        ("GNN", "global"): (render_cached_global_attribution, display_gnn_global),
        ("GNN", "local"): (render_cached_local_attribution, display_gnn_local),
        ("TML", "global"): (render_cached_global_shap_attribution, display_tml_global),
        ("TML", "local"): (render_cached_local_shap_attribution, display_tml_local),
    }[family, scope]
    try:
        parameters = inspect.signature(renderer).parameters
        results = renderer(**{k: v for k, v in args.items() if k in parameters})
        if scope == "local":
            display(results, local_prediction_breakdowns(analysis, selected, args["model_names"]))
        else:
            display(results)
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Unable to plot the saved analysis: {exc}")
    finally:
        plt.close("all")


def _plot_controls(args: dict, family: str, scope: str, prefix: str) -> None:
    with st.expander("Plot settings", expanded=True):
        args["normalisation_type"] = st.radio(
            "Normalisation",
            [
                ImportanceNormalisationMethod.Local,
                ImportanceNormalisationMethod.PerModel,
                ImportanceNormalisationMethod.Global,
                ImportanceNormalisationMethod.NoNormalisation,
            ],
            index=1,
            horizontal=True,
            key=f"{prefix}_normalisation",
            help="PerModel divides each model/bootstrap by its largest absolute attribution across the saved analysis, before averaging. Local scales individual explanations; Global uses one divisor; no normalisation shows raw scores.",
        )
        columns = st.columns(2)
        args["neg_color"] = columns[0].color_picker(
            "Negative attribution colour", "#40bcde", key=f"{prefix}_negative"
        )
        args["pos_color"] = columns[1].color_picker(
            "Positive attribution colour", "#e64747", key=f"{prefix}_positive"
        )
        if scope == "global":
            if family == "GNN":
                args["plot_type"] = st.radio(
                    "Plot type",
                    [AttributionPlotType.Ridge, AttributionPlotType.Bar, AttributionPlotType.Strip],
                    format_func=lambda value: {
                        AttributionPlotType.Ridge: "Ridge (full distribution)",
                        AttributionPlotType.Bar: "Bar (mean ± 95 % CI)",
                        AttributionPlotType.Strip: "Strip (individual scores + mean)",
                    }[value],
                    key=f"{prefix}_style",
                )
                args["top_n"] = int(
                    st.number_input(
                        "Fragments to show (top N + bottom N)",
                        min_value=1,
                        max_value=50,
                        value=10,
                        step=1,
                        key=f"{prefix}_topn",
                    )
                )
            else:
                args["plot_type"] = st.radio(
                    "Plot type",
                    [
                        ShapGlobalPlotType.Beeswarm,
                        ShapGlobalPlotType.Bar,
                        ShapGlobalPlotType.Violin,
                    ],
                    format_func=lambda value: {
                        ShapGlobalPlotType.Beeswarm: "Beeswarm (per-sample, coloured by feature value)",
                        ShapGlobalPlotType.Bar: "Bar (mean |SHAP|)",
                        ShapGlobalPlotType.Violin: "Violin (per-feature distribution)",
                    }[value],
                    key=f"{prefix}_style",
                )
                args["top_n"] = int(
                    st.number_input(
                        "Features to show (top N)",
                        min_value=1,
                        max_value=100,
                        value=10,
                        step=1,
                        key=f"{prefix}_topn",
                    )
                )
        else:
            args["aggregation"] = st.radio(
                "Models display",
                [ExplanationAggregation.Average, ExplanationAggregation.Separate],
                format_func=lambda value: (
                    "Average across models (one plot per sample)"
                    if value == ExplanationAggregation.Average
                    else "Show each model separately (one plot per model)"
                ),
                horizontal=True,
                key=f"{prefix}_aggregation",
            )
            if family == "TML":
                args["local_plot_type"] = st.radio(
                    "Local plot style",
                    ["waterfall", "force", "bar"],
                    horizontal=True,
                    key=f"{prefix}_style",
                )
                args["max_display"] = int(
                    st.number_input(
                        "Features to show (top N)",
                        min_value=1,
                        max_value=100,
                        value=15,
                        step=1,
                        key=f"{prefix}_topn",
                        help="Waterfall and bar use this limit; force shows all features.",
                    )
                )
            else:
                names = {}
                with st.expander("Set custom monomer names (optional)"):
                    for mol in args["dataset"]:
                        labels = []
                        for i, smiles in enumerate(mol.mols):
                            label = st.text_input(
                                f"Name for monomer {i + 1} in {mol.idx}",
                                key=f"{prefix}_name_{mol.idx}_{i}",
                                help=smiles,
                            )
                            labels.append(label or f"Monomer {i + 1}")
                        if any(label != f"Monomer {i + 1}" for i, label in enumerate(labels)):
                            names[mol.idx] = labels
                args["mol_names"] = names
