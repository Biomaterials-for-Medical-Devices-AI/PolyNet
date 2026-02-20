"""
polynet.explainability.visualization
======================================
Plotting utilities for GNN explainability results.

All functions return ``matplotlib.figure.Figure`` objects and have no
Streamlit dependency — rendering and display are handled by the app layer.

Functions
---------
- ``get_cmap`` — custom blue→white→red colormap for attribution heatmaps
- ``plot_mols_with_weights`` — similarity map heatmap over atom attributions
- ``plot_mols_with_numeric_weights`` — attributions as numeric atom labels
- ``plot_attribution_distribution`` — violin-style KDE per fragment/feature
- ``plot_projection_embeddings`` — 2D scatter of dimensionality-reduced embeddings
"""

from __future__ import annotations

import math
from io import BytesIO

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from scipy.stats import gaussian_kde


# ---------------------------------------------------------------------------
# Colormap
# ---------------------------------------------------------------------------


def get_cmap(
    neg_color: str = "#40bcde", pos_color: str = "#e64747"
) -> mcolors.LinearSegmentedColormap:
    """
    Create a custom diverging colormap for attribution heatmaps.

    Transitions from ``neg_color`` through white to ``pos_color``,
    making negative and positive attributions visually distinct without
    the harsh saturation of the default ``bwr`` map.

    Parameters
    ----------
    neg_color:
        Hex colour for negative attributions. Default: soft blue.
    pos_color:
        Hex colour for positive attributions. Default: soft red.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
    """
    return mcolors.LinearSegmentedColormap.from_list(
        "attribution_cmap", [neg_color, "white", pos_color]
    )


# ---------------------------------------------------------------------------
# Molecular attribution plots
# ---------------------------------------------------------------------------


def plot_mols_with_weights(
    smiles_list: list[str],
    weights_list: list[list[float]],
    grid_size: tuple[int, int] | None = None,
    cbar: bool = True,
    legend: list[str] | None = None,
    min_weight: float | None = None,
    max_weight: float | None = None,
    colormap="coolwarm",
) -> plt.Figure:
    """
    Plot molecules with per-atom attribution weights as a similarity heatmap.

    Parameters
    ----------
    smiles_list:
        SMILES strings for the molecules to render.
    weights_list:
        Per-atom attribution weights, one list per molecule. Each list
        must have the same length as the number of atoms in the
        corresponding molecule.
    grid_size:
        ``(rows, cols)`` layout. Inferred from ``len(smiles_list)`` if
        not provided.
    cbar:
        Whether to add a colour bar legend.
    legend:
        Title string for each subplot.
    min_weight, max_weight:
        Explicit colour scale limits. If not provided, derived
        symmetrically from the data range.
    colormap:
        Matplotlib colormap name passed to ``SimilarityMaps``.

    Returns
    -------
    plt.Figure
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]

    for mol, weights in zip(mols, weights_list):
        assert mol is not None, "Invalid SMILES string."
        assert (
            len(weights) == mol.GetNumAtoms()
        ), f"Weight count ({len(weights)}) must match atom count ({mol.GetNumAtoms()})."

    rows, cols = _grid_shape(len(mols), grid_size)
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 3, rows * 3), gridspec_kw={"right": 0.9 if cbar else 1.0}
    )
    axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else [axes]

    if min_weight is None or max_weight is None:
        vmin = min(min(w) for w in weights_list)
        vmax = max(max(w) for w in weights_list)
    else:
        vmin, vmax = min_weight, max_weight

    max_abs = max(abs(vmin), abs(vmax))
    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(vmin=-max_abs, vmax=max_abs)

    for i, (mol, weights) in enumerate(zip(mols, weights_list)):
        AllChem.Compute2DCoords(mol)

        d2d = Draw.MolDraw2DCairo(600, 600)
        Draw.SetACS1996Mode(d2d.drawOptions(), Draw.MeanBondLength(mol))
        opts = d2d.drawOptions()
        opts.clearBackground = True
        opts.addStereoAnnotation = True
        opts.addAtomIndices = False
        opts.bondLineWidth = 2

        SimilarityMaps.GetSimilarityMapFromWeights(
            mol=mol, weights=weights, draw2d=d2d, colorMap=colormap, alpha=0.7, contourLines=10
        )
        d2d.FinishDrawing()
        image = Image.open(BytesIO(d2d.GetDrawingText()))

        axes[i].imshow(image, aspect="auto")
        axes[i].axis("off")
        if legend and i < len(legend):
            axes[i].set_title(legend[i], fontsize=12, pad=10)

    for j in range(len(mols), len(axes)):
        axes[j].axis("off")

    if cbar:
        cbar_ax = fig.add_axes([0.91, 0.3, 0.015, 0.4])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax, label="Attribution")
        cb.ax.tick_params(labelsize=10)

    plt.tight_layout(rect=[0, 0, 0.9 if cbar else 1, 1])
    return fig


def plot_mols_with_numeric_weights(
    smiles_list: list[str],
    weights_list: list[list[float]],
    grid_size: tuple[int, int] | None = None,
    legend: list[str] | None = None,
    decimals: int = 2,
) -> plt.Figure:
    """
    Plot molecules with per-atom attribution weights as numeric labels.

    Parameters
    ----------
    smiles_list:
        SMILES strings for the molecules to render.
    weights_list:
        Per-atom attribution weights.
    grid_size:
        ``(rows, cols)`` layout. Inferred if not provided.
    legend:
        Title string for each subplot.
    decimals:
        Decimal places shown for each weight label.

    Returns
    -------
    plt.Figure
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]

    for mol, weights in zip(mols, weights_list):
        assert mol is not None, "Invalid SMILES string."
        assert (
            len(weights) == mol.GetNumAtoms()
        ), f"Weight count ({len(weights)}) must match atom count ({mol.GetNumAtoms()})."

    rows, cols = _grid_shape(len(mols), grid_size)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, (mol, weights) in enumerate(zip(mols, weights_list)):
        AllChem.Compute2DCoords(mol)

        drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)
        opts = drawer.drawOptions()
        opts.clearBackground = True
        opts.addStereoAnnotation = True
        opts.addAtomIndices = False
        opts.bondLineWidth = 2

        for idx, weight in enumerate(weights):
            opts.atomLabels[idx] = f"{weight:.{decimals}f}"

        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        image = Image.open(BytesIO(drawer.GetDrawingText()))

        axes[i].imshow(image, aspect="auto")
        axes[i].axis("off")
        if legend and i < len(legend):
            axes[i].set_title(legend[i], fontsize=12, pad=10)

    for j in range(len(mols), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Attribution distribution plot
# ---------------------------------------------------------------------------


def plot_attribution_distribution(
    attribution_dict: dict[str, list[float]],
    figsize: tuple[int, int] = (8, 10),
    neg_color: str = "#1f77b4",
    pos_color: str = "#ff7f0e",
    kde_bandwidth: float = 0.2,
    show_averages: bool = True,
) -> plt.Figure:
    """
    Plot per-fragment attribution distributions as violin-style KDE curves.

    Positive and negative attributions are plotted on opposing sides of
    zero, coloured distinctly, with mean markers overlaid.

    Parameters
    ----------
    attribution_dict:
        Mapping from fragment SMILES (or feature name) to list of
        attribution scores across all molecules and occurrences.
    figsize:
        Figure size ``(width, height)`` in inches.
    neg_color, pos_color:
        Matplotlib colour strings for negative and positive sides.
    kde_bandwidth:
        Bandwidth parameter for ``scipy.stats.gaussian_kde``.
    show_averages:
        Whether to overlay mean markers with value annotations.

    Returns
    -------
    plt.Figure
    """
    rcParams.update(
        {
            "font.size": 12,
            "font.family": "sans-serif",
            "axes.linewidth": 1.2,
            "axes.labelsize": 14,
            "axes.titleweight": "bold",
            "axes.titlepad": 15,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
        }
    )

    labels = list(attribution_dict.keys())
    values = list(attribution_dict.values())

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.32, right=0.95)

    y_pos = np.arange(len(labels))
    ax.axvline(0, color="black", linewidth=1.2, linestyle="-", alpha=0.8, zorder=1)

    verts_pos: list = []
    verts_neg: list = []

    for i, vals in enumerate(values):
        vals = np.array(vals, dtype=float)
        if len(vals) == 0:
            continue
        if len(vals) > 1 and np.allclose(vals, vals[0]):
            vals = vals + np.random.normal(0, 1e-6, size=len(vals))

        pos_vals = vals[vals > 0]
        if len(pos_vals) > 1:
            try:
                kde = gaussian_kde(pos_vals, bw_method=kde_bandwidth)
                x = np.linspace(0, max(3, vals.max() * 1.4), 120)
                d = kde(x)
                d = d / (d.max() + 1e-8) * 0.45
                verts_pos.append(list(zip(x, i + d)) + [(0, i)])
            except np.linalg.LinAlgError:
                pass

        neg_vals = vals[vals < 0]
        if len(neg_vals) > 1:
            try:
                kde = gaussian_kde(neg_vals, bw_method=kde_bandwidth)
                x = np.linspace(min(-3, vals.min() * 1.4), 0, 120)
                d = kde(x)
                d = d / (d.max() + 1e-8) * 0.45
                verts_neg.append(list(zip(x, i - d)) + [(0, i)])
            except np.linalg.LinAlgError:
                pass

    if verts_pos:
        ax.add_collection(
            PolyCollection(
                verts_pos,
                facecolors=pos_color,
                edgecolors=pos_color,
                linewidths=1,
                alpha=0.6,
                zorder=3,
            )
        )

    if verts_neg:
        ax.add_collection(
            PolyCollection(
                verts_neg,
                facecolors=neg_color,
                edgecolors=neg_color,
                linewidths=1,
                alpha=0.6,
                zorder=3,
            )
        )

    if show_averages:
        for i, vals in enumerate(values):
            if not len(vals):
                continue
            avg = float(np.mean(vals))
            colour = pos_color if avg > 0 else neg_color
            ax.scatter(avg, i, s=70, color=colour, edgecolors="black", linewidths=0.8, zorder=5)
            ax.text(
                avg + 0.12 * np.sign(avg),
                i,
                f"{avg:.2f}",
                fontsize=11,
                va="center",
                ha="left" if avg > 0 else "right",
                weight="bold",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Attribution", fontsize=14)
    ax.set_title("Attribution Distributions by Functional Group")

    limit = max(3, float(np.abs(ax.get_xlim()).max()))
    ax.set_xlim(-limit, limit)

    ax.grid(True, axis="x", linestyle="--", alpha=0.3, linewidth=0.8, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)

    return fig


# ---------------------------------------------------------------------------
# Embedding projection plot
# ---------------------------------------------------------------------------


def plot_projection_embeddings(
    reduced_embeddings: np.ndarray,
    labels: pd.Series | list | None = None,
    style: pd.Series | list | None = None,
    cmap: str = "blues",
    color_by_name: str | None = None,
    title: str = "Projection of Graph Embeddings",
) -> plt.Figure:
    """
    Scatter plot of 2D dimensionality-reduced graph embeddings.

    For continuous label data, a colour bar replaces the legend.
    For categorical label data, a legend is used.

    Parameters
    ----------
    reduced_embeddings:
        Array of shape ``(n_samples, 2)`` as returned by ``reduce_embeddings``.
    labels:
        Colour mapping. If numeric and high cardinality, treated as continuous.
    style:
        Marker style grouping (passed to seaborn ``style``).
    cmap:
        Matplotlib colormap name.
    color_by_name:
        Label for the colour bar axis (used in continuous mode).
    title:
        Plot title.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=400)

    sns.scatterplot(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        hue=labels,
        style=style,
        palette=cmap,
        s=50,
        ax=ax,
    )

    ax.set_title(title, fontsize=25)
    ax.set_xlabel("Component 1", fontsize=22)
    ax.set_ylabel("Component 2", fontsize=22)
    ax.grid(True)

    label_array = np.array(labels) if labels is not None else None
    is_continuous = (
        label_array is not None
        and np.issubdtype(label_array.dtype, np.number)
        and len(np.unique(label_array)) > 10
    )

    if is_continuous:
        norm = Normalize(vmin=label_array.min(), vmax=label_array.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=color_by_name)
        legend = ax.get_legend()
        if legend:
            legend.remove()

    return fig


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _grid_shape(n: int, grid_size: tuple[int, int] | None) -> tuple[int, int]:
    """Return ``(rows, cols)`` for a grid containing ``n`` items."""
    if grid_size is not None:
        return grid_size
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols
