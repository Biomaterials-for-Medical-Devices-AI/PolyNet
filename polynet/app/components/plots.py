import json

import pandas as pd
import streamlit as st

from polynet.app.options.file_paths import (
    gnn_model_metrics_file_path,
    gnn_plots_directory,
    ml_gnn_results_file_path,
)


def display_predictions(predictions_df: pd.DataFrame):
    """
    Display the predictions in the UI.

    Args:
        predictions_path (Path): The path to the predictions file.
    """
    st.write("### Predictions")
    st.write(predictions_df)


def display_plots(plots_path):

    for plot_file in plots_path.glob("*.png"):
        # Display each plot file in the directory
        st.image(plot_file, use_container_width=True)


def display_model_metrics(metrics_dict):
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


def display_mean_std_model_metrics(metrics_dict):
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
                row[metric] = f"{mean:.3f} ± {std:.3f}"
            else:
                row[metric] = f"{mean:.3f}"
        final_data.append(row)

    final_df = pd.DataFrame(final_data)
    st.subheader("📈 Mean ± Std Model Performance")
    st.dataframe(final_df)


def display_model_results(experiment_path, expanded):

    with st.expander("Model Results", expanded=expanded):

        predictions_path = ml_gnn_results_file_path(
            experiment_path=experiment_path, file_name="predictions.csv"
        )

        if predictions_path.exists():
            predictions = pd.read_csv(
                ml_gnn_results_file_path(
                    experiment_path=experiment_path, file_name="predictions.csv"
                ),
                index_col=0,
            )
            display_predictions(predictions)

        metrics_path = gnn_model_metrics_file_path(experiment_path=experiment_path)
        if metrics_path.exists():
            with open(metrics_path, "rb") as f:
                # Load the metrics dictionary from the file
                metrics_dict = json.load(f)
            display_model_metrics(metrics_dict)
            display_mean_std_model_metrics(metrics_dict)

        plots_path = gnn_plots_directory(experiment_path=experiment_path)
        if plots_path.exists():
            display_plots(plots_path)
