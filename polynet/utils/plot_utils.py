import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from polynet.app.utils import significance_marker
import pandas as pd


def plot_parity(
    y_true,
    y_pred,
    dpi=300,
    height=6,
    width=6,
    title=None,
    title_fontsize=16,
    x_label="True Values",
    x_label_fontsize=14,
    y_label="Predicted Values",
    y_label_fontsize=14,
    tick_size=12,
    grid=True,
    hue=None,
    style_by=None,
    point_color="steelblue",
    border_color="black",
    point_size=50,
    palette=sns.color_palette(),
    legend: bool = True,
    legend_fontsize: int = 12,
):
    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)

    # Plot the diagonal (perfect prediction line)
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

    # Plot the predicted vs. true values
    sns.scatterplot(
        x=y_true,
        y=y_pred,
        ax=ax,
        color=point_color,
        edgecolor=border_color,
        s=point_size,
        hue=hue,
        style=style_by,
        palette=palette,
    )

    # Set labels and styling
    ax.set_xlabel(x_label, fontsize=x_label_fontsize, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=y_label_fontsize, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=tick_size)

    if legend:
        ax.legend(fontsize=legend_fontsize)
    else:
        ax.get_legend().remove()

    if grid:
        ax.grid(True, linestyle="--", alpha=0.7)

    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold")

    fig.tight_layout()  # Adjust layout to prevent clipping of labels

    return fig


def plot_confusion_matrix(
    y_true,
    y_pred,
    dpi=300,
    height=6,
    width=6,
    title=None,
    title_fontsize=16,
    x_label="Predicted label",
    x_label_fontsize=14,
    y_label="True label",
    y_label_fontsize=14,
    label_fontsize=16,
    cmap="Blues",
    tick_size=12,
    display_labels=None,
):

    disp = ConfusionMatrixDisplay(
        confusion_matrix(y_true=y_true, y_pred=y_pred), display_labels=display_labels
    )

    # Customize the figure size
    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)  # Adjust figure size if needed
    disp.plot(ax=ax, cmap=cmap, values_format="d")  # Use 'd' for integer values

    # Increase label font size
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=tick_size)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=tick_size)

    # Increase number (value) font size
    for text in ax.texts:
        text.set_fontsize(label_fontsize)  # Increase number size
        text.set_fontweight("bold")  # Make numbers bold

    ax.set_ylabel(y_label, fontsize=y_label_fontsize, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=x_label_fontsize, fontweight="bold")

    if title:
        plt.title(title, fontsize=title_fontsize, fontweight="bold")

    return fig


def show_label_distribution(data, target_variable, title=None):
    plt.figure(figsize=(8, 6), dpi=300)
    ax = sns.countplot(
        data=data, x=target_variable, hue=target_variable, legend=False, palette="Blues"
    )
    plt.title(title if title else "Label Distribution", fontsize=16)
    plt.xlabel("Labels", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.0f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height() * 0.9),
            ha="center",
            va="center",
            fontsize=12,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )

    return plt


def plot_auroc(
    y_true,
    y_scores,
    dpi=300,
    height=6,
    width=6,
    title=None,
    title_fontsize=16,
    x_label="False Positive Rate",
    x_label_fontsize=14,
    y_label="True Positive Rate",
    y_label_fontsize=14,
    tick_size=12,
    grid=True,
    legend: bool = True,
    legend_fontsize: int = 12,
):

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_scores)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    roc_display.plot(ax=ax)

    if plt.title:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold")

    ax.set_xlabel(x_label, fontsize=x_label_fontsize, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=y_label_fontsize, fontweight="bold")

    ax.tick_params(axis="both", which="major", labelsize=tick_size)

    if grid:
        ax.grid(True, linestyle="--", alpha=0.7)

    if legend:
        ax.legend(fontsize=legend_fontsize)
    else:
        ax.get_legend().remove()

    return fig


def plot_pvalue_matrix(
    p_matrix,
    model_names=None,
    alpha=0.05,
    decimals=3,
    dpi=300,
    height=11,
    width=9,
    title=None,
    title_fontsize=25,
    x_label_fontsize=20,
    y_label_fontsize=20,
    tick_size=15,
    signifficant_colour="#D73027",
    non_signifficant_colour="#4575B4",
    mask_upper_triangle=True,
):
    """
    Plot a lower-triangular McNemar/Wilcoxon p-value matrix with significance colors and stars.

    Parameters
    ----------
    p_matrix : np.ndarray
        Symmetric matrix of p-values (n_models x n_models).
    model_names : list of str, optional
        Names of the models.
    alpha : float
        Significance threshold for coloring.
    decimals : int
        Decimal places to display.
    """
    n = p_matrix.shape[0]
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(n)]

    # Round for annotation display
    annot_matrix = np.empty_like(p_matrix, dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                annot_matrix[i, j] = ""  # leave diagonal empty
            else:
                p = p_matrix[i, j]
                if p < 0.001:
                    annot_matrix[i, j] = significance_marker(p) + "\n<0.001"
                else:
                    annot_matrix[i, j] = f"{significance_marker(p)}\n{p:.{decimals}f}"

    # Mask the upper triangle
    if mask_upper_triangle:
        mask = np.triu(np.ones_like(p_matrix, dtype=bool))
    else:
        mask = np.zeros_like(p_matrix, dtype=bool)

    np.fill_diagonal(mask, True)

    # Colors: red for significant, blue for non-significant
    significant_color = signifficant_colour  # red
    non_significant_color = non_signifficant_colour  # blue
    cmap = sns.color_palette([significant_color, non_significant_color])

    # Plot
    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)

    sns.heatmap(
        p_matrix,
        mask=mask,
        cmap=cmap,
        annot=annot_matrix,
        fmt="",
        linewidths=0.5,
        cbar=False,  # no colorbar (since we're using binary colors)
        vmin=0,
        vmax=1,
        center=alpha,
        annot_kws={"fontsize": tick_size},
        xticklabels=model_names,
        yticklabels=model_names,
        ax=ax,
    )

    # Adjust fonts
    ax.set_xticklabels(model_names, fontsize=x_label_fontsize, rotation=45, ha="right")
    ax.set_yticklabels(model_names, fontsize=y_label_fontsize, rotation=0)

    if title is None:
        title = "Pairwise Model Comparison (p-values)"
    plt.title(title, fontsize=title_fontsize)

    plt.tight_layout()

    return fig


def plot_bootstrap_boxplots(
    metrics_dict,
    metric_name="Metric",
    height=9,
    width=7.5,
    title=None,
    title_fontsize=25,
    x_label_fontsize=20,
    y_label_fontsize=20,
    tick_size=15,
    fill_colour="lightgray",
    border_colour="black",
    median_colour="red",
    dpi=350,
):
    """
    Plot boxplots of bootstrap metric distributions for multiple models,
    with overlayed mean ± 95% CI.

    Parameters
    ----------
    metrics_dict : dict
        {model_name: list/array of bootstrap metric values}
    metric_name : str, optional
        Name of the metric to display on the y-axis (default="Metric").
    title : str, optional
        Title of the plot.
    """
    model_names = list(metrics_dict.keys())
    data = [np.array(vals) for vals in metrics_dict.values()]

    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)

    # Draw boxplots
    plt.boxplot(
        data,
        labels=model_names,
        patch_artist=True,
        boxprops=dict(facecolor=fill_colour, color=border_colour),
        medianprops=dict(color=median_colour, linewidth=2),
    )

    plt.ylabel(metric_name, fontdict={"fontsize": y_label_fontsize})
    ax.tick_params(axis="y", which="major", labelsize=tick_size)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")  # <-- important
        label.set_fontsize(x_label_fontsize)

    title = f"Bootstrap Distributions of {metric_name}" if title is None else title
    plt.title(title, fontsize=title_fontsize)

    plt.tight_layout()
    return fig
