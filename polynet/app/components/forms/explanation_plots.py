"""Page 6 controls for rendering completed explanations, without inference."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from polynet.app.components.forms.fragment_plots import fragment_exports, saved_fragment_table
from polynet.app.services.cache import file_version
from polynet.app.services.explain_model import display_gnn_global, display_gnn_local
from polynet.app.services.explain_source import list_external_datasets, resolve_explain_source
from polynet.app.services.explain_tml import display_tml_global, display_tml_local
from polynet.app.services.explanation_results import (
    analysis_label,
    analysis_summary,
    attribution_coverage,
    attribution_set_membership,
    list_saved_analyses,
    load_analysis,
    local_prediction_breakdowns,
    select_attribution_coverage,
)
from polynet.config.constants import DataSet
from polynet.config.display_names import prettify_label
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
    chosen = _selection_controls(analysis, prefix)
    if not chosen:
        return
    requested_names, coverage = chosen
    selected = sorted(coverage)
    if scope == "local":
        selected = _multiselect(
            "Molecules to display",
            sorted(coverage),
            default=[sorted(coverage)[0]],
            key=f"{prefix}_molecules",
        )
        if not selected:
            st.info("Select at least one explained molecule.")
            return
        for sid in selected:
            names = sorted(coverage[sid])
            st.caption(
                f"{sid}: showing cached explanations from "
                + (", ".join(names) or "none of the selected combinations")
            )
            missing = set(requested_names) - coverage[sid]
            if missing:
                st.info(
                    f"{sid}: {', '.join(sorted(missing))} has no saved score for the chosen set. This plot uses only the combinations listed above. Run page 5 for additional scores."
                )
        if family == "TML":
            st.caption(
                "PerModel uses each model/bootstrap's largest absolute attribution across the whole saved analysis. Normalised plots show scaled model output."
            )
            if args["problem_type"] == "classification":
                st.caption(
                    "SHAP output scales can differ by model (probability or margin/log-odds). Use Separate model display to interpret individual endpoints."
                )
    elif st.checkbox("Manually refine molecule selection", key=f"{prefix}_refine"):
        selected = _multiselect(
            "Molecules to display",
            sorted(coverage),
            default=sorted(coverage),
            key=f"{prefix}_population_molecules",
        )
        if not selected:
            st.info("Select at least one explained molecule.")
            return
    # Both renderers and prediction panels must use exactly the same saved pairs.
    args["model_names"] = sorted(set().union(*(coverage[sid] for sid in selected)))
    allowed = {
        name: {sid for sid in selected if name in coverage[sid]} for name in args["model_names"]
    }
    if family == "GNN":
        stored_ids = {str(mol.idx): mol.idx for mol in args["dataset"]}
        args["explain_mols"] = [stored_ids[sid] for sid in selected]
        args["mols_per_model"] = allowed
        args["dataset"] = [mol for mol in args["dataset"] if str(mol.idx) in selected]
    else:
        args["explain_sample_ids"] = selected
        args["samples_per_model"] = allowed
    st.caption(
        f"Showing {len(selected)} explained molecules · {len(args['model_names'])} model/bootstrap combinations: "
        + ", ".join(args["model_names"])
    )
    _plot_controls(args, family, scope, prefix + "_" + scope)
    renderer, display = {
        ("GNN", "global"): (render_cached_global_attribution, display_gnn_global),
        ("GNN", "local"): (render_cached_local_attribution, display_gnn_local),
        ("TML", "global"): (render_cached_global_shap_attribution, display_tml_global),
        ("TML", "local"): (render_cached_local_shap_attribution, display_tml_local),
    }[family, scope]
    try:
        if family == "GNN" and scope == "global":
            fragment_table = saved_fragment_table(analysis)
            labels_column = "Tag" if args["fragment_label_mode"] == "F tags" else "SMARTS"
            args["fragment_labels"] = dict(
                zip(fragment_table["Fragment"], fragment_table[labels_column])
            )
        parameters = inspect.signature(renderer).parameters
        results = renderer(**{k: v for k, v in args.items() if k in parameters})
        if scope == "local":
            display(
                results,
                local_prediction_breakdowns(analysis, selected, args["model_names"], allowed),
            )
        else:
            display(results)
            if family == "GNN" and args["fragment_label_mode"] == "F tags" and not results.warning:
                fragment_exports(results, fragment_table, args["top_n"], prefix + "_fragments")
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Unable to plot the saved analysis: {exc}")
    finally:
        plt.close("all")


def _multiselect(label, options, default, key, **kwargs):
    """Keep selections valid when an upstream architecture/set choice changes."""
    if key in st.session_state:
        st.session_state[key] = [value for value in st.session_state[key] if value in options]
    return st.multiselect(label, options, default=default, key=key, **kwargs)


def _selection_controls(analysis: dict, prefix: str):
    """Page 5-style selection in either view, limited to completed scores."""
    all_names = sorted(set().union(*attribution_coverage(analysis).values()))
    architectures = sorted({name.rsplit("_", 1)[0] for name in all_names})
    selected_architectures = _multiselect(
        "GNN architectures to include" if analysis["family"] == "GNN" else "Models to include",
        architectures,
        default=architectures,
        format_func=prettify_label,
        key=f"{prefix}_architectures",
    )
    # Keep bootstrap options stable across architecture changes; only real saved
    # combinations enter the plot, even for incomplete architecture/split grids.
    iterations = sorted({int(name.rsplit("_", 1)[1]) for name in all_names})
    selected_iterations = _multiselect(
        "Bootstrap iterations to include",
        iterations,
        default=iterations,
        key=f"{prefix}_bootstraps",
    )
    names = [
        name
        for name in all_names
        if name.rsplit("_", 1)[0] in selected_architectures
        and int(name.rsplit("_", 1)[1]) in selected_iterations
    ]
    if st.checkbox("Choose individual model/bootstrap combinations", key=f"{prefix}_exact"):
        names = _multiselect(
            "Cached models / bootstraps to show", names, default=names, key=f"{prefix}_models"
        )
    if not names:
        st.info("Select at least one saved model/bootstrap combination.")
        return None
    membership = attribution_set_membership(analysis)
    saved_sets = {
        value for entries in membership.values() for name, value in entries.items() if name in names
    }
    options = ["All cached molecules"] + [
        value
        for value in (DataSet.Training, DataSet.Validation, DataSet.Test)
        if value in saved_sets
    ]
    if len(options) > 1:
        key = f"{prefix}_set"
        if st.session_state.get(key) not in options:
            st.session_state[key] = options[0]
        set_name = st.selectbox("Explained set", options, key=key)
    else:
        set_name = options[0]
    st.caption(
        "Only scores saved in this analysis are available. Set membership is evaluated separately for each bootstrap; use page 5 to calculate additional scores."
    )
    coverage = select_attribution_coverage(
        analysis, names, None if set_name == options[0] else set_name
    )
    if not coverage:
        st.info("No saved explanations match these models, bootstraps and set.")
        return None
    return names, coverage


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
                args["fragment_label_mode"] = st.radio(
                    "Fragment labels",
                    ["SMARTS", "F tags"],
                    horizontal=True,
                    key=f"{prefix}_fragment_labels",
                    help="Display RDKit SMARTS or fixed F1, F2, … tags. Labels never change cached identities or attribution scores.",
                )
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
