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
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


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
