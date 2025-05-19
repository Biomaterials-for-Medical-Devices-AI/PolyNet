import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(
    y_true, y_pred, display_labels=None, title=None, show=False, save_path=None
):
    disp = ConfusionMatrixDisplay(
        confusion_matrix(y_true=y_true, y_pred=y_pred),
        display_labels=display_labels,
    )

    # Customize the figure size
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)  # Adjust figure size if needed
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d")  # Use 'd' for integer values

    # Increase label font size
    ax.set_xticklabels(
        ax.get_xticklabels(),
        fontsize=12,
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        fontsize=12,
    )

    # Increase number (value) font size
    for text in ax.texts:
        text.set_fontsize(16)  # Increase number size
        text.set_fontweight("bold")  # Make numbers bold

    ax.set_ylabel("True label", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted label", fontsize=14, fontweight="bold")

    if title:
        plt.title(title, fontsize=16, fontweight="bold")

    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()
    plt.clf()
