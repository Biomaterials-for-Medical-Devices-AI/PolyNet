"""Page 6 controls for rendering completed explanations, without inference."""

from __future__ import annotations

from datetime import datetime
import hashlib
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from polynet.app.services.explain_model import display_gnn_global, display_gnn_local
from polynet.app.services.explain_source import list_external_datasets, resolve_explain_source
from polynet.app.services.explain_tml import display_tml_global, display_tml_local
from polynet.app.services.explanation_results import (
    list_cached_analyses,
    list_saved_analyses,
    load_analysis,
    load_cached_analysis,
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
    """Load a completed run, then redraw it with the existing presentation options."""
    st.subheader("Feature importance plots")
    sources = {"Experiment data": experiment_path}
    for name in list_external_datasets(experiment_path):
        sources[f"External dataset: {name}"] = resolve_explain_source(
            experiment_path, name
        ).cache_root
    available = {
        label: root
        for label, root in sources.items()
        if list_saved_analyses(root)
        or (root / "explanations" / "explanation.json").exists()
        or any((root / "explanations").glob("shap_*.csv"))
    }
    if not available:
        st.info(
            "No feature importance analysis found. Run a global or local explanation on page 5 — Explain Models first."
        )
        return
    if not st.checkbox("Customise feature importance plots", key="analyse_explanations_enabled"):
        return
    label = st.selectbox(
        "Explanation data source", list(available), key="analyse_explanation_source"
    )
    root = available[label]
    snapshots = list_saved_analyses(root)
    use_cache = not snapshots or st.checkbox(
        "Load an older GUI / CLI attribution cache", key="analyse_explanation_legacy"
    )
    loaded_key = f"loaded_explanation_{experiment_path}"
    if use_cache:
        try:
            entries = list_cached_analyses(root)
        except (OSError, ValueError, KeyError) as exc:
            st.error(f"Could not read saved attribution caches: {exc}")
            return
        if not entries:
            st.info("No compatible attribution values found. Run the analysis again on page 5.")
            return
        selected = st.selectbox(
            "Cached feature importance analysis",
            range(len(entries)),
            format_func=lambda i: f"{entries[i]['family']} — class {entries[i]['class_key']}"
            + (f" — {entries[i]['fragmentation']}" if entries[i]["family"] == "GNN" else " — SHAP"),
            key="analyse_explanation_cache",
        )
        scope = st.radio(
            "Explanation view", ["global", "local"], key="analyse_explanation_view", horizontal=True
        )
        entry = entries[selected]
        version = hashlib.sha256(repr(entry["cache_version"]).encode()).hexdigest()[:12]
        identity = f"{root}:cache:{entry['family']}:{entry['class_key']}:{entry.get('fragmentation', '')}:{scope}:{version}"
        st.caption(
            "Older caches do not record the original sample/set selection. This view uses all cached samples and models for the selected class and fragmentation. No missing scores will be calculated."
        )
        loader = lambda: load_cached_analysis(entries[selected], root, scope, data_options)
    else:
        path = st.selectbox(
            "Completed feature importance analysis",
            snapshots,
            format_func=lambda p: f"{p.stem.split('_')[0].upper()} — {p.stem.split('_')[1]} — {datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')} ({p.stem.rsplit('_', 1)[1][:6]})",
            key="analyse_explanation_snapshot",
        )
        identity = f"{path}:{path.stat().st_mtime_ns}"
        loader = lambda: load_analysis(path)
    if st.button("Load selected analysis", key="analyse_explanation_load"):
        try:
            st.session_state[loaded_key] = (identity, loader())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            st.error(f"Could not load this analysis: {exc}. Run it again on page 5.")
            return
    loaded = st.session_state.get(loaded_key)
    if loaded is None or loaded[0] != identity:
        st.info("Load the selected analysis to customise its plots.")
        return
    analysis = loaded[1]
    args = analysis["arguments"].copy()
    family, scope = analysis["family"], analysis["scope"]
    prefix = "explanation_plot_" + hashlib.sha256(identity.encode()).hexdigest()[:12]
    ids = args.get("explain_mols", args.get("explain_sample_ids", []))
    st.caption(
        f"{family} {scope} analysis · {len(ids)} selected samples · {len(args['model_names'])} trained model instances"
        + (f" · completed {analysis['created_at']}" if "created_at" in analysis else "")
    )
    st.write("Models: " + ", ".join(args["model_names"]))
    if args.get("target_class") is not None:
        st.caption(f"Target class: {args['target_class']}")
    _plot_controls(args, family, scope, prefix)
    renderer, display = {
        ("GNN", "global"): (render_cached_global_attribution, display_gnn_global),
        ("GNN", "local"): (render_cached_local_attribution, display_gnn_local),
        ("TML", "global"): (render_cached_global_shap_attribution, display_tml_global),
        ("TML", "local"): (render_cached_local_shap_attribution, display_tml_local),
    }[family, scope]
    parameters = inspect.signature(renderer).parameters
    try:
        # Rendering APIs accept raw scores only: no models, training or cache writes.
        results = renderer(**{k: v for k, v in args.items() if k in parameters})
        if scope == "local":
            display(results, analysis.get("prediction_breakdowns"))
        else:
            display(results)
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Unable to plot the saved analysis: {exc}")
    finally:
        # Streamlit has serialized the figures; release their pyplot references.
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
            help="Local scales individual explanations; PerModel scales each trained model; Global uses one divisor; no normalisation shows raw scores. Local plot normalisation retains the existing explanation behaviour.",
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
