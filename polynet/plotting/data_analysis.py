import matplotlib.pyplot as plt
import seaborn as sns


def show_label_distribution(data, target_variable, title=None, class_names=None):
    # If class_names dict is provided, map the target variable to human-readable names
    if class_names:
        label_column = f"{target_variable}_name"
        data = data.copy()
        data[label_column] = data[target_variable].astype(str).map(class_names)
        x_axis = label_column
        hue_axis = label_column
    else:
        x_axis = target_variable
        hue_axis = target_variable

    fig, ax = plt.subplots(figsize=(8, 6), dpi=500)
    ax = sns.countplot(data=data, x=x_axis, hue=hue_axis, legend=False, palette="Blues")

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

    return fig


def show_continuous_distribution(data, target_variable, bins=30, title=None):
    """
    Plot the distribution of a continuous target variable.

    Args:
        data (pd.DataFrame): The dataset containing the target variable.
        target_variable (str): Column name of the continuous variable.
        bins (int): Number of bins in the histogram.
        title (str, optional): Plot title.
        show (bool): Whether to display the plot.
        save_path (str, optional): Path to save the plot as an image.
    """

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax = sns.histplot(
        data=data, x=target_variable, bins=bins, kde=True, color="skyblue", edgecolor="black"
    )
    plt.title(title if title else "Value Distribution", fontsize=14)
    plt.xlabel("Values", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Annotate bin heights
    for patch in ax.patches:
        height = patch.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}",
                (patch.get_x() + patch.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
            )

    return fig
