"""
polynet.app.services.graph_viz
================================
Build an interactive Plotly figure from a PyG ``Data`` object produced by
``CustomPolymerGraph``.

The figure shows the *actual* graph that was built and stored on disk — not a
schematic of what was expected.  Node and edge features are decoded from the
tensors using the same ordering that ``_atom_features`` / ``_bond_features``
use when encoding, so what you see in the hover tooltip is exactly what the
GNN receives as input.

Layout
------
Atoms are positioned using RDKit 2-D coordinates (stripped of PSMILES wildcard
atoms when needed).  For multi-monomer polymers each sub-graph is offset
horizontally.  When RDKit fails a circular fallback layout is used.

Hover
-----
- **Atom node** → element symbol, monomer index, all decoded node features.
- **Bond midpoint** → decoded edge features (same for both directions since
  ``CustomPolymerGraph`` stores both directions with the same feature vector).

Bond line width is scaled by decoded bond order when ``GetBondTypeAsDouble``
is among the selected features.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from polynet.config.enums import AtomBondDescriptorDictKey, AtomFeature, BondFeature


# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

_ATOM_COLORS: Dict[str, str] = {
    "C": "#404040", "N": "#3050F8", "O": "#FF0D0D", "H": "#AAAAAA",
    "F": "#90E050", "Cl": "#1FF01F", "Br": "#A62929", "I": "#940094",
    "S": "#FFFF30", "P": "#FF8000", "Si": "#F0C8A0", "B": "#FFB5B5",
    "*": "#888888",   # PSMILES wildcard / attachment-point placeholder
}
_DEFAULT_ATOM_COLOR = "#909090"

# Border colours per monomer — makes sub-graphs visually distinct
_MONOMER_BORDERS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]

_BOND_WIDTH_BY_ORDER = {1.0: 2, 1.5: 3, 2.0: 4, 3.0: 5}
_DEFAULT_BOND_WIDTH = 2


# ---------------------------------------------------------------------------
# Feature-order declaration
# Must stay in sync with CustomPolymerGraph._atom_features / _bond_features.
#
# IMPORTANT — why the user's config order does NOT matter here:
# ``_atom_features`` iterates a hard-coded ``feature_extractors`` dict (Python
# 3.7+ insertion-order semantics) and only appends a feature when it is present
# in ``self.node_feats``.  The *user's* config dict is only consulted for
# presence and per-feature settings — the position in the config dict is
# irrelevant to the encoded tensor layout.  ``_bond_features`` uses the same
# pattern with explicit ``if BondFeature.X in self.edge_feats`` guards.
# Therefore these decoder lists must mirror the encoder's hard-coded order,
# NOT the user-defined config order.  If the encoder ever changes its ordering,
# this list must be updated in lock-step.
# ---------------------------------------------------------------------------

_ATOM_FEATURE_ORDER: List[AtomFeature] = [
    AtomFeature.GetAtomicNum,
    AtomFeature.GetTotalNumHs,
    AtomFeature.GetTotalDegree,
    AtomFeature.GetImplicitValence,
    AtomFeature.GetIsAromatic,
    AtomFeature.GetHybridization,
    AtomFeature.GetFormalCharge,
    AtomFeature.GetChiralTag,
    AtomFeature.GetMass,
    AtomFeature.IsInRing,
    AtomFeature.IsAttachmentPoint,
]

_BOND_FEATURE_ORDER: List[BondFeature] = [
    BondFeature.GetBondTypeAsDouble,
    BondFeature.GetStereo,
    BondFeature.IsInRing,
    BondFeature.GetIsConjugated,
    BondFeature.GetIsAromatic,
]

# Scalar (single-column) features — everything else is one-hot encoded.
_ATOM_SCALARS = {
    AtomFeature.GetIsAromatic,
    AtomFeature.GetMass,
    AtomFeature.IsInRing,
    AtomFeature.IsAttachmentPoint,
}
_BOND_SCALARS = {
    BondFeature.IsInRing,
    BondFeature.GetIsConjugated,
    BondFeature.GetIsAromatic,
}


# ---------------------------------------------------------------------------
# Feature decoding
# ---------------------------------------------------------------------------

def _feature_width(feat, config: dict, scalar_set: set) -> int:
    """Number of tensor columns used by one feature."""
    if feat in scalar_set:
        return 1
    vals = config.get(AtomBondDescriptorDictKey.AllowableVals, [])
    wildcard = config.get(AtomBondDescriptorDictKey.Wildcard, False)
    return len(vals) + (1 if wildcard else 0)


def _decode_one(feat, vec: torch.Tensor, config: dict, scalar_set: set) -> str:
    """Decode a single feature sub-vector to a display string."""
    if vec.numel() == 0:
        return "—"
    if feat in scalar_set:
        val = vec[0].item()
        if feat == AtomFeature.GetMass:
            return f"{val * 100:.2f} Da"
        return "Yes" if val > 0.5 else "No"
    vals = config.get(AtomBondDescriptorDictKey.AllowableVals, [])
    wildcard = config.get(AtomBondDescriptorDictKey.Wildcard, False)
    hot = int(vec.argmax().item())
    if hot < len(vals):
        return str(vals[hot])
    return "other" if wildcard else "—"


def decode_node_features(
    x: torch.Tensor,
    node_feats_config: Dict,
) -> List[Dict[str, str]]:
    """Decode the node feature tensor to one display-dict per atom."""
    active = [
        (f, node_feats_config[f])
        for f in _ATOM_FEATURE_ORDER
        if f in node_feats_config
    ]
    result: List[Dict[str, str]] = []
    for n in range(x.shape[0]):
        col = 0
        d: Dict[str, str] = {}
        for feat, cfg in active:
            w = _feature_width(feat, cfg, _ATOM_SCALARS)
            if w == 0:
                continue  # feature contributed 0 columns (empty AllowableVals); skip
            d[str(feat)] = _decode_one(feat, x[n, col: col + w], cfg, _ATOM_SCALARS)
            col += w
        result.append(d)
    return result


def decode_edge_features(
    edge_attr: torch.Tensor,
    edge_feats_config: Dict,
) -> List[Dict[str, str]]:
    """Decode the edge feature tensor to one display-dict per directed edge."""
    active = [
        (f, edge_feats_config[f])
        for f in _BOND_FEATURE_ORDER
        if f in edge_feats_config
    ]
    result: List[Dict[str, str]] = []
    for e in range(edge_attr.shape[0]):
        col = 0
        d: Dict[str, str] = {}
        for feat, cfg in active:
            w = _feature_width(feat, cfg, _BOND_SCALARS)
            if w == 0:
                continue  # feature contributed 0 columns (empty AllowableVals); skip
            d[str(feat)] = _decode_one(feat, edge_attr[e, col: col + w], cfg, _BOND_SCALARS)
            col += w
        result.append(d)
    return result


# ---------------------------------------------------------------------------
# 2-D layout helpers
# ---------------------------------------------------------------------------

def _strip_dummy_atoms(mol: Chem.Mol) -> Chem.Mol:
    """Remove wildcard (atomic-num 0) atoms — mirrors _strip_wildcards logic."""
    dummy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0}
    if not dummy:
        return mol
    rw = Chem.RWMol(mol)
    for idx in sorted(dummy, reverse=True):
        rw.RemoveAtom(idx)
    return rw.GetMol()


def _rdkit_2d_coords(smiles: str, expected_n: int) -> Optional[np.ndarray]:
    """RDKit 2-D coordinates for *smiles*.

    Uses a two-pass strategy to handle both PSMILES configurations:

    - **Pass 1 — full SMILES**: the graph tensor includes ``*`` atoms because
      ``IsAttachmentPoint`` was *not* selected, so the featuriser left wildcard
      atoms as ordinary nodes.
    - **Pass 2 — stripped SMILES**: ``IsAttachmentPoint`` *was* selected so
      ``_strip_wildcards`` already removed ``*`` atoms before featurisation;
      we strip here too so the counts agree.

    Returns ``None`` when parsing fails or neither pass matches *expected_n*.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Pass 1: use the full molecule (preserves * atoms in the layout)
    if mol.GetNumAtoms() != expected_n:
        # Pass 2: mirror what the featuriser did when IsAttachmentPoint was on
        mol = _strip_dummy_atoms(mol)
        if mol.GetNumAtoms() != expected_n:
            return None
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    return np.array(
        [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
         for i in range(mol.GetNumAtoms())]
    )


def _circular_layout(n: int, radius: float = 1.0) -> np.ndarray:
    """Equally-spaced nodes on a circle — fallback layout."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _build_positions(data: Data) -> np.ndarray:
    """2-D positions for every atom in the graph.

    Each monomer sub-graph is positioned independently (using RDKit or the
    circular fallback) then offset horizontally so sub-graphs don't overlap.
    """
    n_atoms = data.x.shape[0]
    positions = np.zeros((n_atoms, 2))
    monomer_id = (
        data.monomer_id.numpy().flatten()
        if data.monomer_id is not None
        else np.zeros(n_atoms, dtype=int)
    )
    mols_smiles: List[str] = data.mols if hasattr(data, "mols") else []

    x_offset = 0.0
    for mon_idx, smiles in enumerate(mols_smiles):
        mask = np.where(monomer_id == mon_idx)[0]
        n = len(mask)
        if n == 0:
            continue

        pos = _rdkit_2d_coords(smiles, n)
        if pos is None:
            pos = _circular_layout(n)

        # Normalise to a fixed scale so all sub-graphs are comparably sized
        span = max((pos.max(axis=0) - pos.min(axis=0)).max(), 1e-6)
        pos = pos / span
        pos -= pos.mean(axis=0)       # centre
        pos[:, 0] += x_offset         # horizontal offset

        positions[mask] = pos
        x_offset += (pos[:, 0].max() - pos[:, 0].min()) + 1.5   # gap between monomers

    return positions


def _atom_symbols(data: Data) -> List[str]:
    """Element symbol for each graph node, read directly from the SMILES."""
    n_atoms = data.x.shape[0]
    symbols = ["?"] * n_atoms
    monomer_id = (
        data.monomer_id.numpy().flatten()
        if data.monomer_id is not None
        else np.zeros(n_atoms, dtype=int)
    )
    mols_smiles: List[str] = data.mols if hasattr(data, "mols") else []

    for mon_idx, smiles in enumerate(mols_smiles):
        mask = np.where(monomer_id == mon_idx)[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        # Pass 1: full SMILES — * atoms are graph nodes (IsAttachmentPoint off)
        if mol.GetNumAtoms() != len(mask):
            # Pass 2: strip wildcards — featuriser removed them (IsAttachmentPoint on)
            mol = _strip_dummy_atoms(mol)
        if mol.GetNumAtoms() != len(mask):
            continue
        for local_i, global_i in enumerate(mask):
            symbols[global_i] = mol.GetAtomWithIdx(local_i).GetSymbol()

    return symbols


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------

def build_graph_figure(
    data: Data,
    node_feats_config: Dict,
    edge_feats_config: Dict,
    monomer_filter: Optional[int] = None,
) -> go.Figure:
    """Return an interactive Plotly figure of the graph stored in *data*.

    Parameters
    ----------
    data:
        PyG ``Data`` object produced by ``CustomPolymerGraph``.
    node_feats_config:
        ``repr_cfg.node_features`` — the same dict that drove featurisation.
    edge_feats_config:
        ``repr_cfg.edge_features`` — the same dict that drove featurisation.
    monomer_filter:
        When set to a monomer index (0, 1, …) only that monomer's atoms and
        bonds are drawn and the view is recentred on that sub-graph.  ``None``
        (default) draws all monomers side-by-side.
    """
    n_atoms = data.x.shape[0]
    edge_index = data.edge_index.numpy()   # (2, 2*n_bonds)
    monomer_id = (
        data.monomer_id.numpy().flatten()
        if data.monomer_id is not None
        else np.zeros(n_atoms, dtype=int)
    )
    mols_smiles: List[str] = data.mols if hasattr(data, "mols") else []

    positions = _build_positions(data)
    symbols = _atom_symbols(data)
    node_decoded = decode_node_features(data.x, node_feats_config)
    edge_decoded = (
        decode_edge_features(data.edge_attr, edge_feats_config)
        if data.edge_attr is not None
        else [{}] * edge_index.shape[1]
    )

    # ------------------------------------------------------------------
    # When a single monomer is selected, recentre the view on it so the
    # sub-graph fills the plot area instead of being offset to the right.
    # ------------------------------------------------------------------
    if monomer_filter is not None:
        vis_mask = np.where(monomer_id == monomer_filter)[0]
        if len(vis_mask) > 0:
            center = positions[vis_mask].mean(axis=0)
            positions = positions - center

    traces: list = []
    unique_monomers = sorted(set(monomer_id.tolist()))
    # Monomers that will actually be drawn in this call
    visible_monomers = (
        [monomer_filter]
        if monomer_filter is not None and monomer_filter in unique_monomers
        else unique_monomers
    )
    show_legend = len(unique_monomers) > 1 and monomer_filter is None

    # ------------------------------------------------------------------
    # Edge traces
    # Each bond is stored twice (both directions). Draw once (i < j).
    # ------------------------------------------------------------------
    seen: set = set()
    for e in range(edge_index.shape[1]):
        i, j = int(edge_index[0, e]), int(edge_index[1, e])
        if i >= j or (i, j) in seen:
            continue
        # When filtering, skip bonds that cross monomer boundaries or
        # belong to a different monomer.
        if monomer_filter is not None and monomer_id[i] != monomer_filter:
            continue
        seen.add((i, j))

        xi, yi = positions[i]
        xj, yj = positions[j]
        mid_x, mid_y = (xi + xj) / 2, (yi + yj) / 2

        # Infer bond width from decoded bond order
        bond_type_str = edge_decoded[e].get(str(BondFeature.GetBondTypeAsDouble), "1.0")
        try:
            line_width = _BOND_WIDTH_BY_ORDER.get(float(bond_type_str), _DEFAULT_BOND_WIDTH)
        except ValueError:
            line_width = _DEFAULT_BOND_WIDTH

        hover_lines = [f"<b>Bond  {symbols[i]} – {symbols[j]}</b>"]
        hover_lines += [f"&nbsp;&nbsp;{k}: {v}" for k, v in edge_decoded[e].items()]

        # Bond line (hover disabled — line targets are too thin to click reliably)
        traces.append(go.Scatter(
            x=[xi, xj, None], y=[yi, yj, None],
            mode="lines",
            line=dict(width=line_width, color="#555555"),
            hoverinfo="skip",
            showlegend=False,
        ))

        # Invisible midpoint marker that carries the hover tooltip
        traces.append(go.Scatter(
            x=[mid_x], y=[mid_y],
            mode="markers",
            marker=dict(size=12, color="rgba(0,0,0,0)"),
            hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
            showlegend=False,
        ))

    # ------------------------------------------------------------------
    # Node traces — one per monomer for a clean legend
    # ------------------------------------------------------------------
    for mon_idx in visible_monomers:
        mask = np.where(monomer_id == mon_idx)[0]
        if len(mask) == 0:
            continue

        node_x = positions[mask, 0].tolist()
        node_y = positions[mask, 1].tolist()
        syms = [symbols[i] for i in mask]
        fill_colors = [_ATOM_COLORS.get(s, _DEFAULT_ATOM_COLOR) for s in syms]
        border_color = _MONOMER_BORDERS[mon_idx % len(_MONOMER_BORDERS)]

        # Display index is 1-based to match the monomer selector in the UI
        display_idx = mon_idx + 1

        hover_texts = []
        for global_i in mask:
            lines = [f"<b>Atom {global_i}  ({symbols[global_i]})</b>"]
            if len(unique_monomers) > 1:
                label = mols_smiles[mon_idx] if mon_idx < len(mols_smiles) else f"Monomer {display_idx}"
                lines.append(f"Monomer {display_idx}: {label[:45]}{'…' if len(label) > 45 else ''}")
            lines += [f"&nbsp;&nbsp;{k}: {v}" for k, v in node_decoded[global_i].items()]
            hover_texts.append("<br>".join(lines) + "<extra></extra>")

        legend_smiles = mols_smiles[mon_idx] if mon_idx < len(mols_smiles) else f"Monomer {display_idx}"
        legend_label = (
            f"Monomer {display_idx}: {legend_smiles[:35]}{'…' if len(legend_smiles) > 35 else ''}"
        )

        traces.append(go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            marker=dict(
                size=32,
                color=fill_colors,
                line=dict(color=border_color, width=3),
            ),
            text=syms,
            textposition="middle center",
            textfont=dict(size=11, color="white", family="Arial Black"),
            hovertemplate=hover_texts,
            name=legend_label,
            showlegend=show_legend,
        ))

    # Taller figure when a single monomer is isolated — gives more room
    # to spread out the atoms without them overlapping.
    fig_height = 600 if monomer_filter is not None else 540

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            showlegend=show_legend,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, constrain="domain"),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"),
            margin=dict(b=10, l=10, r=10, t=10),
            height=fig_height,
            plot_bgcolor="white",
            paper_bgcolor="white",
        ),
    )
    return fig
