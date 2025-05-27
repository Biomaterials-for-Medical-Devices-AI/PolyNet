import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(y_true, y_pred, display_labels=None, title=None):
    disp = ConfusionMatrixDisplay(
        confusion_matrix(y_true=y_true, y_pred=y_pred), display_labels=display_labels
    )

    # Customize the figure size
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)  # Adjust figure size if needed
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d")  # Use 'd' for integer values

    # Increase label font size
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

    # Increase number (value) font size
    for text in ax.texts:
        text.set_fontsize(16)  # Increase number size
        text.set_fontweight("bold")  # Make numbers bold

    ax.set_ylabel("True label", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted label", fontsize=14, fontweight="bold")

    if title:
        plt.title(title, fontsize=16, fontweight="bold")

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
