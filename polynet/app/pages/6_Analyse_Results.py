import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from polynet.app.components.experiments import experiment_selector
from polynet.app.components.figure_export import display_exportable_figure
from polynet.app.components.forms.analyse_results import (
    compare_metrics_form,
    compare_predictions_form,
    confusion_matrix_plot_form,
    parity_plot_form,
)
from polynet.app.components.forms.explanation_plots import explanation_plots_form
from polynet.app.components.plots import display_model_results
from polynet.app.options.file_paths import (
    data_options_path,
    general_options_path,
    ml_results_file_path,
    model_metrics_file_path,
    polynet_experiments_base_dir,
    representation_options_path,
    split_indices_path,
    train_gnn_model_options_path,
    train_tml_model_options_path,
)
from polynet.app.services.cache import read_csv, read_json
from polynet.app.services.configurations import load_options
from polynet.app.services.experiments import get_experiments
from polynet.config.enums import ProblemType
from polynet.config.schemas import (
    DataConfig,
    GeneralConfig,
    RepresentationConfig,
    SplitConfig,
    TrainGNNConfig,
    TrainTMLConfig,
)
from polynet.utils.statistical_analysis import mean_split_sizes

st.header("Analyse Results")

st.markdown(
    """
    In this section, you can analyse the results of the models you have trained, for both
    graph neural network (GNN) and traditional machine learning (TML) frameworks. Select an
    experiment to explore its predictions and metrics. Depending on the problem type, you can
    inspect parity plots (regression) or confusion matrices (classification) per model, set,
    and iteration, and statistically compare models against each other on their predictions
    and metrics.
    """
)


choices = get_experiments()
experiment_name = experiment_selector(choices)


if experiment_name:

    experiment_path = polynet_experiments_base_dir() / experiment_name

    path_to_data_opts = data_options_path(
        experiment_path=polynet_experiments_base_dir() / experiment_name
    )

    data_options = load_options(path=path_to_data_opts, options_class=DataConfig)

    path_to_representation_opts = representation_options_path(
        experiment_path=polynet_experiments_base_dir() / experiment_name
    )
    representation_options = load_options(
        path=path_to_representation_opts, options_class=RepresentationConfig
    )

    # GNN and TML training options are loaded only when present, so experiments
    # that trained only one framework (or both) all work on this page.
    path_to_train_gnn_options = train_gnn_model_options_path(experiment_path=experiment_path)
    train_gnn_options = (
        load_options(path=path_to_train_gnn_options, options_class=TrainGNNConfig)
        if path_to_train_gnn_options.exists()
        else None
    )

    path_to_train_tml_options = train_tml_model_options_path(experiment_path=experiment_path)
    train_tml_options = (
        load_options(path=path_to_train_tml_options, options_class=TrainTMLConfig)
        if path_to_train_tml_options.exists()
        else None
    )

    path_to_general_opts = general_options_path(experiment_path=experiment_path)
    general_experiment_options = load_options(
        path=path_to_general_opts, options_class=GeneralConfig
    )

    path_to_split_options = experiment_path / "split_options.json"
    split_options = load_options(path=path_to_split_options, options_class=SplitConfig)
    split_type = split_options.split_type

    if (
        not (path_to_train_gnn_options.exists() or path_to_train_tml_options.exists())
        or not path_to_general_opts.exists()
    ):
        st.error("No models have been trained yet. Please train a GNN or TML model first.")
        st.stop()

    display_model_results(experiment_path=experiment_path, expanded=False, export_plots=True)

    predictions_path = ml_results_file_path(experiment_path=experiment_path)
    predictions = read_csv(predictions_path, index_col=0)
    metrics_path = model_metrics_file_path(experiment_path=experiment_path)
    metrics = read_json(metrics_path)

    if st.checkbox("Show predictions data"):
        st.dataframe(predictions)

    if data_options.problem_type == ProblemType.Regression:

        st.subheader("Parity Plot")

        parity_plot = parity_plot_form(
            predictions_df=predictions, split_type=split_type, data_options=data_options
        )

        if parity_plot:

            display_exportable_figure(
                parity_plot,
                export_options={
                    "dpi": int(parity_plot.dpi),
                    "prefix": f"results_export_{experiment_name}",
                },
                plot_id=("parity",),
            )

            if st.button("Save Parity Plot"):
                parity_plot_path = experiment_path / "parity_plot.png"
                parity_plot.savefig(parity_plot_path)
                st.success(f"Parity plot saved to {parity_plot_path}")
            plt.close(parity_plot)

    elif data_options.problem_type == ProblemType.Classification:
        st.subheader("Confusion Matrix")

        confusion_matrix_plot = confusion_matrix_plot_form(
            predictions_df=predictions, split_type=split_type, data_options=data_options
        )

        if confusion_matrix_plot:
            display_exportable_figure(
                confusion_matrix_plot,
                export_options={
                    "dpi": int(confusion_matrix_plot.dpi),
                    "prefix": f"results_export_{experiment_name}",
                },
                plot_id=("confusion-matrix",),
            )
            plt.close(confusion_matrix_plot)

    st.subheader("Compare models")

    compare_preds_plot = compare_predictions_form(
        predictions_df=predictions,
        target_variable_name=data_options.target_variable_name,
        data_options=data_options,
    )

    if compare_preds_plot:

        display_exportable_figure(
            compare_preds_plot,
            export_options={
                "dpi": int(compare_preds_plot.dpi),
                "prefix": f"results_export_{experiment_name}",
            },
            plot_id=("prediction-p-values",),
        )
        plt.close(compare_preds_plot)

    st.divider()

    splits_file = split_indices_path(experiment_path=experiment_path)
    split_sizes = None
    if splits_file.exists():
        split_sizes = mean_split_sizes(read_json(splits_file))

    compare_metrics_plot = compare_metrics_form(
        metrics=metrics, data_options=data_options, split_sizes=split_sizes
    )

    if compare_metrics_plot:
        display_exportable_figure(
            compare_metrics_plot,
            export_options={
                "dpi": int(compare_metrics_plot.dpi),
                "prefix": f"results_export_{experiment_name}",
            },
            plot_id=("metric-comparison",),
        )
        plt.close(compare_metrics_plot)

    st.divider()
    explanation_plots_form(experiment_path, data_options)
