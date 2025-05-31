import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


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
