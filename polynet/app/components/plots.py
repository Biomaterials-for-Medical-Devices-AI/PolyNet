from collections import defaultdict
from pathlib import Path
import re

import pandas as pd
import streamlit as st

from polynet.app.components.images import show_image_full_width
from polynet.app.options.file_paths import (
    ml_results_file_path,
    model_metrics_file_path,
    plots_directory,
    unseen_predictions_parent_path,
)
from polynet.app.services.cache import read_csv, read_json


def display_predictions(predictions_df: pd.DataFrame):
    """
    Display the predictions in the UI.

    Args:
        predictions_path (Path): The path to the predictions file.
    """
    st.write("### Predictions")
    st.write(predictions_df)


def display_plots(plots_path: Path, *, export_plots: bool = False):
    """
    Organize and display plots from a directory with the following hierarchy:
    - Model name
      - Iteration (e.g., fold or bootstrap)
        - Plot type (confusion matrix, AUROC, parity plot)
    """
    plot_groups = defaultdict(lambda: defaultdict(list))

    for plot_file in plots_path.glob("*.png"):
        name = plot_file.stem  # filename without extension

        # Try to match all types
        if match := re.match(r"(.+?)_(\d+)_confusion_matrix", name):
            model, iteration = match.groups()
            plot_type = "Confusion Matrix"

        elif match := re.match(r"(.+?)_(\d+)_class_(\d+)_roc_curve", name):
            model, iteration, class_num = match.groups()
            plot_type = f"AUROC (Class {class_num})"

        elif match := re.match(r"(.+?)_(\d+)_parity_plot", name):
            model, iteration = match.groups()
            plot_type = "Parity Plot"

        elif match := re.match(r"(.+?)_(\d+)_learning_curve", name):
            model, iteration = match.groups()
            plot_type = "Learning Curve"

        else:
            continue  # Unrecognized format

        plot_groups[model][int(iteration)].append((plot_type, plot_file))

    # Sort and display
    for model in sorted(plot_groups):
        st.markdown(f"### Model: `{model}`")
        for iteration in sorted(plot_groups[model]):
            st.markdown(f"**Iteration {iteration}**")
            for plot_type, plot_path in sorted(plot_groups[model][iteration]):
                st.markdown(f"*{plot_type}*")
                show_image_full_width(plot_path)
                if export_plots:
                    st.download_button(
                        "Export plot",
                        data=plot_path.read_bytes(),
                        file_name=plot_path.name,
                        mime="image/png",
                        key=f"saved_plot_export:{plot_path.resolve()}",
                        help="Downloads the original saved PNG without resizing or recompression.",
                        on_click="ignore",
                    )


def display_model_metrics(metrics_dict: dict):
    """
    Display model performance metrics in a Streamlit dataframe.

    Parameters:
    - metrics_dict: dict of the form {
          1: {
              'ModelA': {
                  'Training': {'MAE': 0.1, 'RMSE': 0.2},
                  'Validation': {'MAE': 0.15, 'RMSE': 0.25},
              },
              'ModelB': {
                  ...
              },
          },
          2: {
              ...
          }
      }
    """
    records = []

    for iteration, models in metrics_dict.items():
        for model_name, sets in models.items():
            for set_name, metrics in sets.items():
                record = {"Model": model_name, "Iteration": iteration, "Set": set_name, **metrics}
                records.append(record)

    df = pd.DataFrame(records, index=None)
    df = df.round(3)

    st.subheader("📊 Model Performance Metrics")
    st.dataframe(df)


def display_mean_std_model_metrics(metrics_dict: dict):
    """
    Display the mean ± std of model performance metrics per model per set across iterations.
    If only one value exists, std is omitted.
    """
    records = []

    # Flatten nested dict into list of records
    for iteration, models in metrics_dict.items():
        for model_name, sets in models.items():
            for set_name, metrics in sets.items():
                record = {"Model": model_name, "Set": set_name, **metrics}
                records.append(record)

    df = pd.DataFrame(records)

    # Group by Model and Set
    grouped = df.groupby(["Model", "Set"])
    stats = grouped.agg(["mean", "std", "count"]).round(3)

    # Flatten the column MultiIndex
    final_data = []
    metric_names = df.columns.difference(["Model", "Set"])

    for (model, set_name), group_stats in stats.iterrows():
        row = {"Model": model, "Set": set_name}
        for metric in metric_names:
            mean = group_stats[(metric, "mean")]
            std = group_stats[(metric, "std")]
            count = group_stats[(metric, "count")]
            if count > 1:
                row[metric] = f"{mean:.2f} ± {std:.2f}"
            else:
                row[metric] = f"{mean:.2f}"
        final_data.append(row)

    final_df = pd.DataFrame(final_data)
    st.subheader("📈 Mean ± Std Model Performance")
    st.dataframe(final_df)


def display_model_results(
    experiment_path: Path,
    expanded: bool,
    expander_str: str = "Model Results",
    *,
    external: bool = False,
    export_plots: bool = False,
):
    """Load saved results on demand, using the training or external-run layout.

    External runs write predictions.csv and metrics.json directly in their
    dataset folder; older runs may use the training-style ml_results layout.
    ``export_plots`` adds a download of each original saved PNG on page 6.
    """
    # Collapsed expanders still execute their bodies in Streamlit. An explicit
    # toggle avoids reading and rendering all previous runs on each page visit.
    if not st.toggle(
        f"Show {expander_str}",
        value=expanded,
        key=f"results:{experiment_path.resolve()}:{expander_str}:{expanded}",
    ):
        return

    with st.expander(expander_str, expanded=True):

        direct_layout = external and (
            (experiment_path / "predictions.csv").exists()
            or (experiment_path / "metrics.json").exists()
        )
        predictions_path = (
            experiment_path / "predictions.csv"
            if direct_layout
            else ml_results_file_path(experiment_path=experiment_path)
        )
        displayed = False

        if predictions_path.exists():
            predictions = read_csv(predictions_path, index_col=0)
            display_predictions(predictions)
            displayed = True

        metrics_path = (
            experiment_path / "metrics.json"
            if direct_layout
            else model_metrics_file_path(experiment_path=experiment_path)
        )
        if metrics_path.exists():
            metrics_dict = read_json(metrics_path)
            if metrics_dict:
                display_model_metrics(metrics_dict)
                split_metrics = {k: v for k, v in metrics_dict.items() if k != "ensemble"}
                if split_metrics:
                    display_mean_std_model_metrics(split_metrics)
                displayed = True

        plots_path = (
            experiment_path / "plots"
            if direct_layout
            else plots_directory(experiment_path=experiment_path)
        )
        if plots_path.exists():
            display_plots(plots_path, export_plots=export_plots)
            displayed = displayed or any(plots_path.glob("*.png"))
        if not displayed:
            st.info("No saved predictions, metrics or plots were found for this run.")


def display_unseen_predictions(experiment_path: Path):
    unseen_results_path = unseen_predictions_parent_path(experiment_path=experiment_path)

    if not unseen_results_path.exists():
        return

    subdirs = [p for p in unseen_results_path.iterdir() if p.is_dir()]

    for path in subdirs:
        experiment = path.name

        display_model_results(
            experiment_path=path, expanded=False, expander_str=experiment, external=True
        )
