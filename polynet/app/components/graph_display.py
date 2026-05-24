"""
polynet.app.components.graph_display
======================================
Streamlit component that renders the interactive graph-preview section on the
Representation page.

Design
------
The section reads *processed* ``.pt`` files that ``build_graph_dataset`` wrote
to ``<experiment>/representation/GNN/processed/`` and shows the actual graph
that the GNN will receive — not a schematic.  Because the presence check is
done against disk files (not Streamlit session state), the section is always
visible after the dataset has been built, even after a full page reload or
molecule-selector change.  This fixes the "display disappeared on re-select"
problem: the figure is re-generated in Python on every Streamlit rerun and
passed to ``st.plotly_chart``, which renders it correctly regardless of what
widget triggered the rerun.

Usage
-----
    from polynet.app.components.graph_display import show_graph_visualization

    show_graph_visualization(
        experiment_path=experiment_path,
        data_opts=data_opts,
        repr_cfg=repr_cfg,
    )
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from polynet.app.services.graph_viz import build_graph_figure
from polynet.config.paths import gnn_data_path, gnn_raw_data_path
from polynet.config.schemas.data import DataConfig
from polynet.config.schemas.representation import RepresentationConfig

# Filename prefix used by CustomPolymerGraph (from _graph_filename)
_DATASET_CLASS = "CustomPolymerGraph"


def _processed_dir(experiment_path: Path) -> Path:
    return gnn_data_path(experiment_path) / "processed"


def _pt_path(processed_dir: Path, row_idx: int, target_col: str) -> Path:
    """Reconstruct the .pt path for a given raw-CSV row index."""
    return processed_dir / f"{_DATASET_CLASS}_{row_idx}_{target_col}.pt"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def show_graph_visualization(
    experiment_path: Path, data_opts: DataConfig, repr_cfg: RepresentationConfig
) -> None:
    """Render the graph-preview section.

    Safe to call unconditionally — returns immediately when no graph dataset
    exists yet (e.g. before Apply Representation Settings is first clicked, or
    when only molecular descriptors were configured).

    Parameters
    ----------
    experiment_path:
        Absolute path to the experiment directory.
    data_opts:
        Loaded ``DataConfig`` for this experiment.
    repr_cfg:
        Loaded or current ``RepresentationConfig`` (used to decode features).
    """
    proc_dir = _processed_dir(experiment_path)

    # Nothing to show if the graph dataset hasn't been built yet
    if not proc_dir.exists() or not any(proc_dir.glob("*.pt")):
        return

    st.markdown("---")
    st.markdown("### 🔍 Graph Representation Preview")
    st.markdown(
        "The graph below is the **actual** representation stored on disk for the "
        "selected molecule — not what we expect it to look like, but exactly what "
        "the GNN will receive as input.  "
        "Hover over **nodes** (atoms) or **bond midpoints** to inspect every "
        "encoded feature value."
    )

    # -----------------------------------------------------------------------
    # Molecule selector — reads the raw GNN CSV for IDs (no tensor loading)
    # -----------------------------------------------------------------------
    raw_csv = gnn_raw_data_path(experiment_path) / data_opts.data_name
    if not raw_csv.exists():
        st.warning("Raw GNN CSV not found — cannot list molecules.")
        return

    raw_df = pd.read_csv(raw_csv)
    id_col = data_opts.id_col

    if id_col and id_col in raw_df.columns:
        mol_options = raw_df[id_col].tolist()
    else:
        mol_options = list(raw_df.index)

    selected = st.selectbox(
        "Select molecule to inspect", options=mol_options, key="graph_viz_mol_selector"
    )
    if selected is None:
        return

    # -----------------------------------------------------------------------
    # Resolve raw-CSV row index → .pt filename
    # -----------------------------------------------------------------------
    if id_col and id_col in raw_df.columns:
        hits = raw_df.index[raw_df[id_col] == selected].tolist()
        if not hits:
            st.error(f"Molecule '{selected}' not found in the raw GNN CSV.")
            return
        row_idx = int(hits[0])
    else:
        row_idx = int(selected)

    pt = _pt_path(proc_dir, row_idx, data_opts.target_variable_col)
    if not pt.exists():
        st.warning(
            f"Graph file `{pt.name}` not found. "
            "Click **Apply Representation Settings** to rebuild the dataset."
        )
        return

    # -----------------------------------------------------------------------
    # Load graph & show summary metrics
    # -----------------------------------------------------------------------
    graph_data = torch.load(pt, weights_only=False)

    n_nodes = graph_data.num_nodes
    n_edges = graph_data.edge_index.shape[1] // 2  # bidirectional storage
    n_monomers = len(graph_data.mols) if hasattr(graph_data, "mols") else 1
    node_feat_dim = graph_data.x.shape[1] if graph_data.x is not None else 0
    edge_feat_dim = graph_data.edge_attr.shape[1] if graph_data.edge_attr is not None else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Atoms (nodes)", n_nodes)
    c2.metric("Bonds (edges)", n_edges)
    c3.metric("Monomers", n_monomers)
    c4.metric("Node feat dim", node_feat_dim)
    c5.metric("Edge feat dim", edge_feat_dim)

    # -----------------------------------------------------------------------
    # Monomer selector — lets the user zoom into one sub-graph at a time.
    # Shown only for multi-monomer polymers; single-monomer always shows all.
    # -----------------------------------------------------------------------
    monomer_filter: int | None = None
    if n_monomers > 1:
        mols_list = graph_data.mols if hasattr(graph_data, "mols") else []
        # Labels are 1-based for users ("Monomer 1", "Monomer 2", …).
        # Wrap the SMILES in backticks so Streamlit's Markdown renderer does not
        # interpret * characters as italic/bold markers (e.g. "*CC*" → *CC*).
        monomer_options = ["All monomers"] + [
            (
                f"Monomer {i + 1}: `{mols_list[i][:45]}{'…' if len(mols_list[i]) > 45 else ''}`"
                if i < len(mols_list)
                else f"Monomer {i + 1}"
            )
            for i in range(n_monomers)
        ]
        monomer_sel = st.radio(
            "View", options=monomer_options, horizontal=True, key="graph_viz_monomer_filter"
        )
        if monomer_sel != "All monomers":
            # Parse the 1-based display index and convert back to 0-based internal index
            monomer_filter = int(monomer_sel.split(":")[0].replace("Monomer ", "").strip()) - 1

        if monomer_filter is None:
            st.info(
                f"This polymer contains **{n_monomers} monomers** shown as separate "
                "disconnected sub-graphs. The coloured node border indicates which "
                "monomer each atom belongs to. Select a monomer above to zoom in."
            )

    # -----------------------------------------------------------------------
    # Build and render the interactive Plotly figure
    # -----------------------------------------------------------------------
    fig = build_graph_figure(
        data=graph_data,
        node_feats_config=repr_cfg.node_features,
        edge_feats_config=repr_cfg.edge_features,
        monomer_filter=monomer_filter,
    )
    st.plotly_chart(fig, use_container_width=True)
