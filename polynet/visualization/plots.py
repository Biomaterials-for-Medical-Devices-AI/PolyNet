"""
polynet.visualization.plots
============================
Publication-quality plotting functions for model evaluation and training diagnostics.

All functions return a ``matplotlib.figure.Figure`` and have no side effects
beyond creating the figure — saving is handled separately by
``polynet.visualization.utils.save_plot``.

Functions
---------
- ``plot_learning_curve``   — training / validation / test loss over epochs
- ``plot_parity``           — predicted vs. true values for regression
- ``plot_auroc``            — ROC curve with AUC annotation
- ``plot_confusion_matrix`` — annotated confusion matrix for classification
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from polynet.utils.statistical_analysis import significance_marker

# ---------------------------------------------------------------------------
# Training diagnostics
# ---------------------------------------------------------------------------


def plot_learning_curve(
    losses: tuple[list[float], list[float], list[float]], title: str = "Learning Curve"
) -> plt.Figure:
    """
    Plot training, validation, and test losses over epochs.

    Parameters
    ----------
    losses:
        Tuple of ``(train_losses, val_losses, test_losses)``, where each
        element is a list of per-epoch loss values as stored on
        ``model.losses`` after training.
    title:
        Plot title.

    Returns
    -------
    plt.Figure
    """
    train_losses, val_losses, test_losses = losses
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(epochs, train_losses, label="Training Loss", color="tab:blue", linewidth=2)
    ax.plot(epochs, val_losses, label="Validation Loss", color="tab:orange", linewidth=2)
    ax.plot(epochs, test_losses, label="Test Loss", color="tab:green", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def plot_parity(
    y_true,
    y_pred,
    title: str | None = None,
    hue=None,
    style_by=None,
    point_color: str = "steelblue",
    border_color: str = "black",
    point_size: int = 50,
    palette=None,
    legend: bool = True,
    height: int = 6,
    width: int = 6,
    dpi: int = 300,
    title_fontsize: int = 16,
    x_label: str = "True Values",
    x_label_fontsize: int = 14,
    y_label: str = "Predicted Values",
    y_label_fontsize: int = 14,
    tick_size: int = 12,
    legend_fontsize: int = 12,
    grid: bool = True,
) -> plt.Figure:
    """
    Parity plot (predicted vs. true values) for regression evaluation.

    A dashed diagonal represents the ideal ``y = x`` line. Points are
    plotted with seaborn so optional ``hue`` and ``style_by`` groupings
    are supported.

    Parameters
    ----------
    y_true:
        Ground-truth target values.
    y_pred:
        Model predictions.
    title:
        Optional plot title.
    hue:
        Optional grouping variable for point colour.
    style_by:
        Optional grouping variable for point marker style.
    point_color:
        Default point colour when ``hue`` is not set.
    border_color:
        Point edge colour.
    point_size:
        Scatter point size in points².
    palette:
        Seaborn palette used when ``hue`` is set.
    legend:
        Whether to display the legend.
    height, width:
        Figure dimensions in inches.
    dpi:
        Figure resolution.
    title_fontsize, x_label_fontsize, y_label_fontsize, tick_size, legend_fontsize:
        Font sizes for the respective plot elements.
    grid:
        Whether to show a dashed grid.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        ls="--",
        color="black",
        lw=2,
        label="Parity Line (y=x)",
    )

    sns.scatterplot(
        x=y_true,
        y=y_pred,
        ax=ax,
        color=point_color,
        edgecolor=border_color,
        s=point_size,
        hue=hue,
        style=style_by,
        palette=palette or sns.color_palette(),
    )

    ax.set_xlabel(x_label, fontsize=x_label_fontsize, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=y_label_fontsize, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=tick_size)

    if legend:
        ax.legend(fontsize=legend_fontsize)
    elif ax.get_legend():
        ax.get_legend().remove()

    if grid:
        ax.grid(True, linestyle="--", alpha=0.7)

    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def plot_auroc(
    y_true,
    y_scores,
    title: str | None = None,
    legend: bool = True,
    height: int = 6,
    width: int = 6,
    dpi: int = 300,
    title_fontsize: int = 16,
    x_label: str = "False Positive Rate",
    x_label_fontsize: int = 14,
    y_label: str = "True Positive Rate",
    y_label_fontsize: int = 14,
    tick_size: int = 12,
    legend_fontsize: int = 12,
    grid: bool = True,
) -> plt.Figure:
    """
    ROC curve with AUC score annotation.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels.
    y_scores:
        Predicted probability scores for the positive class.
    title:
        Optional plot title.
    legend:
        Whether to show the AUC legend entry.
    height, width:
        Figure dimensions in inches.
    dpi:
        Figure resolution.
    title_fontsize, x_label_fontsize, y_label_fontsize, tick_size, legend_fontsize:
        Font sizes for the respective plot elements.
    grid:
        Whether to show a dashed grid.

    Returns
    -------
    plt.Figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_scores)

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot(ax=ax)

    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold")

    ax.set_xlabel(x_label, fontsize=x_label_fontsize, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=y_label_fontsize, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=tick_size)

    if grid:
        ax.grid(True, linestyle="--", alpha=0.7)

    if legend:
        ax.legend(fontsize=legend_fontsize)
    elif ax.get_legend():
        ax.get_legend().remove()

    return fig


def plot_confusion_matrix(
    y_true,
    y_pred,
    display_labels=None,
    title: str | None = None,
    cmap: str = "Blues",
    height: int = 6,
    width: int = 6,
    dpi: int = 300,
    title_fontsize: int = 16,
    x_label: str = "Predicted label",
    x_label_fontsize: int = 14,
    y_label: str = "True label",
    y_label_fontsize: int = 14,
    tick_size: int = 12,
    label_fontsize: int = 16,
) -> plt.Figure:
    """
    Annotated confusion matrix for classification evaluation.

    Parameters
    ----------
    y_true:
        Ground-truth class labels.
    y_pred:
        Predicted class labels.
    display_labels:
        Optional list of class name strings for axis tick labels.
    title:
        Optional plot title.
    cmap:
        Matplotlib colormap name for the matrix cells.
    height, width:
        Figure dimensions in inches.
    dpi:
        Figure resolution.
    title_fontsize, x_label_fontsize, y_label_fontsize, tick_size, label_fontsize:
        Font sizes for the respective plot elements.

    Returns
    -------
    plt.Figure
    """
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    disp.plot(ax=ax, cmap=cmap, values_format="d")

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=tick_size)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=tick_size)

    for text in ax.texts:
        text.set_fontsize(label_fontsize)
        text.set_fontweight("bold")

    ax.set_xlabel(x_label, fontsize=x_label_fontsize, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=y_label_fontsize, fontweight="bold")

    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold")

    return fig


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------


def plot_pvalue_matrix(
    p_matrix,
    model_names=None,
    alpha: float = 0.05,
    decimals: int = 3,
    dpi: int = 300,
    height: int = 11,
    width: int = 9,
    title: str | None = None,
    title_fontsize: int = 25,
    x_label_fontsize: int = 20,
    y_label_fontsize: int = 20,
    tick_size: int = 15,
    signifficant_colour: str = "#D73027",
    non_signifficant_colour: str = "#4575B4",
    mask_upper_triangle: bool = True,
) -> plt.Figure:
    """
    Plot a lower-triangular p-value matrix with significance colours and stars.

    Parameters
    ----------
    p_matrix:
        Symmetric matrix of p-values (n_models × n_models).
    model_names:
        Names of the models. Defaults to "Model 1", "Model 2", …
    alpha:
        Significance threshold for colouring.
    decimals:
        Decimal places to display.
    """
    n = p_matrix.shape[0]
    if model_names is None:
        model_names = [f"Model {i + 1}" for i in range(n)]

    annot_matrix = np.empty_like(p_matrix, dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                annot_matrix[i, j] = ""
            else:
                p = p_matrix[i, j]
                if p < 0.001:
                    annot_matrix[i, j] = significance_marker(p) + "\n<0.001"
                else:
                    annot_matrix[i, j] = f"{significance_marker(p)}\n{p:.{decimals}f}"

    if mask_upper_triangle:
        mask = np.triu(np.ones_like(p_matrix, dtype=bool))
    else:
        mask = np.zeros_like(p_matrix, dtype=bool)
    np.fill_diagonal(mask, True)

    cmap = sns.color_palette([signifficant_colour, non_signifficant_colour])

    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    sns.heatmap(
        p_matrix,
        mask=mask,
        cmap=cmap,
        annot=annot_matrix,
        fmt="",
        linewidths=0.5,
        cbar=False,
        vmin=0,
        vmax=1,
        center=alpha,
        annot_kws={"fontsize": tick_size},
        xticklabels=model_names,
        yticklabels=model_names,
        ax=ax,
    )

    ax.set_xticklabels(model_names, fontsize=x_label_fontsize, rotation=45, ha="right")
    ax.set_yticklabels(model_names, fontsize=y_label_fontsize, rotation=0)

    if title is None:
        title = "Pairwise Model Comparison (p-values)"
    ax.set_title(title, fontsize=title_fontsize)

    fig.tight_layout()
    return fig


def plot_bootstrap_boxplots(
    metrics_dict: dict,
    metric_name: str = "Metric",
    height: int = 9,
    width: int = 7,
    title: str | None = None,
    title_fontsize: int = 25,
    x_label_fontsize: int = 20,
    y_label_fontsize: int = 20,
    tick_size: int = 15,
    fill_colour: str = "lightgray",
    border_colour: str = "black",
    median_colour: str = "red",
    dpi: int = 350,
) -> plt.Figure:
    """
    Boxplots of bootstrap metric distributions for multiple models with
    overlaid mean ± 95 % CI.

    Parameters
    ----------
    metrics_dict:
        ``{model_name: list/array of bootstrap metric values}``
    metric_name:
        Y-axis label.
    title:
        Optional plot title. Defaults to "Bootstrap Distributions of {metric_name}".
    """
    model_names = list(metrics_dict.keys())
    data = [np.array(vals) for vals in metrics_dict.values()]

    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    ax.boxplot(
        data,
        labels=model_names,
        patch_artist=True,
        boxprops=dict(facecolor=fill_colour, color=border_colour),
        medianprops=dict(color=median_colour, linewidth=2),
    )

    ax.set_ylabel(metric_name, fontsize=y_label_fontsize)
    ax.tick_params(axis="y", which="major", labelsize=tick_size)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")
        label.set_fontsize(x_label_fontsize)

    ax.set_title(
        f"Bootstrap Distributions of {metric_name}" if title is None else title,
        fontsize=title_fontsize,
    )

    fig.tight_layout()
    return fig
