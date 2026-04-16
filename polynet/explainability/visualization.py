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

from io import BytesIO
import math

from PIL import Image
from matplotlib import rcParams
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
import seaborn as sns

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
    kde_bandwidth: float = 0.5,
    top_n: int | None = None,
) -> plt.Figure:
    """
    Plot per-fragment attribution distributions as an overlapping ridge (joy) plot.

    Each fragment occupies one row of a seaborn ``FacetGrid``; rows overlap
    vertically so the overall chart is compact.  Fragments are sorted by their
    mean attribution (descending), and each row is coloured by interpolating
    between ``neg_color`` (most negative mean) and ``pos_color`` (most positive
    mean).

    Parameters
    ----------
    attribution_dict:
        Mapping from fragment SMILES (or feature name) to list of
        attribution scores across all molecules and occurrences.
    figsize:
        Approximate figure size ``(width, height)`` in inches.  The height is
        used to derive the ``FacetGrid`` row height (``height / n_fragments``).
    neg_color, pos_color:
        Matplotlib colour strings for the tails of the fragment palette.
    kde_bandwidth:
        ``bw_adjust`` parameter forwarded to ``sns.kdeplot``.
    top_n:
        If set, only the ``top_n`` fragments with the highest mean attribution
        and the ``top_n`` with the lowest mean attribution are shown.
        Pass ``None`` to show all fragments.

    Returns
    -------
    plt.Figure
    """
    if not attribution_dict:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No attribution data", ha="center", va="center")
        return fig

    df, ordered_frags, means = _prepare_attribution_df(attribution_dict, top_n)
    n = len(ordered_frags)
    palette = _build_fragment_palette(ordered_frags, means, neg_color, pos_color)

    row_height = max(0.4, figsize[1] / n)
    aspect = figsize[0] / row_height

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    g = sns.FacetGrid(
        df,
        row="fragment",
        hue="fragment",
        aspect=aspect,
        height=row_height,
        palette=palette,
    )

    # Filled KDE
    g.map(
        sns.kdeplot,
        "attribution",
        bw_adjust=kde_bandwidth,
        clip_on=False,
        fill=True,
        alpha=0.8,
        linewidth=1.5,
    )

    # White outline for separation between overlapping rows
    g.map(
        sns.kdeplot,
        "attribution",
        bw_adjust=kde_bandwidth,
        clip_on=False,
        color="w",
        lw=2,
    )

    # Zero baseline
    g.refline(y=0, linewidth=1, linestyle="-", color="black", clip_on=False)

    # Fragment label on the left of each row
    def _label(x, color, label):
        ax = plt.gca()
        ax.text(
            0,
            0.2,
            label,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
        )

    g.map(_label, "attribution")
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.set_xlabels("Attribution")
    g.despine(bottom=True, left=True)

    # Overlap rows
    g.figure.subplots_adjust(hspace=-0.3)

    return g.figure


def _prepare_attribution_df(
    attribution_dict: dict[str, list[float]],
    top_n: int | None,
) -> tuple[pd.DataFrame, list[str], pd.Series]:
    """
    Shared data-preparation step for all global attribution plots.

    Returns
    -------
    df : long-form DataFrame with columns ``fragment`` and ``attribution``
    ordered_frags : fragment names sorted by mean (descending, already filtered)
    means : per-fragment mean Series (filtered)
    """
    rows = []
    for frag, scores in attribution_dict.items():
        for s in scores:
            rows.append({"fragment": frag, "attribution": float(s)})
    df = pd.DataFrame(rows)

    means = df.groupby("fragment")["attribution"].mean()
    ordered_frags = means.sort_values(ascending=False).index.tolist()

    if top_n is not None and len(ordered_frags) > top_n * 2:
        ordered_frags = ordered_frags[:top_n] + ordered_frags[-top_n:]
        df = df[df["fragment"].isin(ordered_frags)]
        means = means[ordered_frags]

    df["fragment"] = pd.Categorical(df["fragment"], categories=ordered_frags, ordered=True)
    df = df.sort_values("fragment")
    return df, ordered_frags, means


def _build_fragment_palette(
    ordered_frags: list[str],
    means: pd.Series,
    neg_color: str,
    pos_color: str,
) -> dict:
    neg_rgb = np.array(mcolors.to_rgb(neg_color))
    pos_rgb = np.array(mcolors.to_rgb(pos_color))
    min_mean, max_mean = means.min(), means.max()
    denom = max_mean - min_mean if max_mean != min_mean else 1.0
    return {
        frag: tuple((neg_rgb + (pos_rgb - neg_rgb) * (means[frag] - min_mean) / denom).tolist())
        for frag in ordered_frags
    }


def plot_attribution_bar(
    attribution_dict: dict[str, list[float]],
    figsize: tuple[int, int] = (10, 6),
    neg_color: str = "#1f77b4",
    pos_color: str = "#ff7f0e",
    top_n: int | None = None,
) -> plt.Figure:
    """
    Horizontal bar chart of mean fragment attributions with 95 % CI error bars.

    Bars are coloured by the same neg→pos interpolation used in the ridge plot,
    sorted from most positive (top) to most negative (bottom).  Error bars show
    the 95 % bootstrap confidence interval across all scores for that fragment.

    Parameters
    ----------
    attribution_dict:
        ``{frag_smiles: [score1, score2, ...]}``.
    figsize:
        Figure size ``(width, height)``.
    neg_color, pos_color:
        Hex colours for negative / positive attribution ends of the palette.
    top_n:
        Show only the ``top_n`` highest and ``top_n`` lowest mean-attribution
        fragments. Pass ``None`` to show all.

    Returns
    -------
    plt.Figure
    """
    if not attribution_dict:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No attribution data", ha="center", va="center")
        return fig

    df, ordered_frags, means = _prepare_attribution_df(attribution_dict, top_n)
    palette = _build_fragment_palette(ordered_frags, means, neg_color, pos_color)

    # 95 % CI via 1.96 × SEM
    stats = df.groupby("fragment", observed=True)["attribution"].agg(["mean", "sem", "count"])
    stats["ci95"] = 1.96 * stats["sem"].fillna(0)

    # Plot in descending mean order (top of chart = most positive)
    plot_frags = list(reversed(ordered_frags))
    y_pos = np.arange(len(plot_frags))
    bar_means = stats.loc[plot_frags, "mean"].values
    bar_ci = stats.loc[plot_frags, "ci95"].values
    colors = [palette[f] for f in plot_frags]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(y_pos, bar_means, xerr=bar_ci, color=colors, height=0.6,
                   error_kw={"ecolor": "0.3", "capsize": 3, "linewidth": 1.2})

    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_frags, fontsize=9)
    ax.set_xlabel("Mean Attribution (Y − Y_masked)", fontsize=11)
    ax.set_title("Fragment Mean Attribution", fontsize=13, pad=10)
    sns.despine(ax=ax, left=True)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_attribution_strip(
    attribution_dict: dict[str, list[float]],
    figsize: tuple[int, int] = (10, 6),
    neg_color: str = "#1f77b4",
    pos_color: str = "#ff7f0e",
    top_n: int | None = None,
) -> plt.Figure:
    """
    Horizontal strip / swarm-style plot showing every individual attribution score.

    Each fragment gets one row; individual scores are jittered vertically so
    overlapping points are visible.  The mean is overlaid as a larger diamond
    marker.  This combines the transparency of the ridge plot (shows spread and
    outliers) with the compactness of the bar chart.

    Parameters
    ----------
    attribution_dict:
        ``{frag_smiles: [score1, score2, ...]}``.
    figsize:
        Figure size ``(width, height)``.
    neg_color, pos_color:
        Hex colours for the palette endpoints.
    top_n:
        Limit to top-N and bottom-N fragments by mean. Pass ``None`` for all.

    Returns
    -------
    plt.Figure
    """
    if not attribution_dict:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No attribution data", ha="center", va="center")
        return fig

    df, ordered_frags, means = _prepare_attribution_df(attribution_dict, top_n)
    palette = _build_fragment_palette(ordered_frags, means, neg_color, pos_color)

    n = len(ordered_frags)
    fig, ax = plt.subplots(figsize=figsize)

    rng = np.random.default_rng(seed=0)
    for i, frag in enumerate(reversed(ordered_frags)):
        scores = df.loc[df["fragment"] == frag, "attribution"].values
        jitter = rng.uniform(-0.25, 0.25, size=len(scores))
        color = palette[frag]
        ax.scatter(scores, np.full_like(scores, i) + jitter, color=color,
                   alpha=0.55, s=18, linewidths=0, zorder=2)
        # Mean marker
        ax.scatter([means[frag]], [i], color=color, s=80, marker="D",
                   edgecolors="0.2", linewidths=0.8, zorder=3)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(list(reversed(ordered_frags)), fontsize=9)
    ax.set_xlabel("Attribution (Y − Y_masked)", fontsize=11)
    ax.set_title("Fragment Attribution — Individual Scores + Mean (◆)", fontsize=13, pad=10)
    sns.despine(ax=ax, left=True)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
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
