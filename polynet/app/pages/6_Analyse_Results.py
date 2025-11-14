import json

import pandas as pd
import streamlit as st

from polynet.app.components.experiments import experiment_selector
from polynet.app.components.forms.analyse_results import (
    compare_metrics_form,
    compare_predictions_form,
    confusion_matrix_plot_form,
    parity_plot_form,
)
from polynet.app.components.plots import display_model_results
from polynet.app.options.data import DataOptions
from polynet.app.options.file_paths import (
    data_options_path,
    general_options_path,
    model_metrics_file_path,
    ml_results_file_path,
    polynet_experiments_base_dir,
    representation_options_path,
    train_gnn_model_options_path,
)
from polynet.app.options.general_experiment import GeneralConfigOptions
from polynet.app.options.representation import RepresentationOptions
from polynet.app.options.train_GNN import TrainGNNOptions
from polynet.app.services.configurations import load_options
from polynet.app.services.experiments import get_experiments
from polynet.options.enums import ProblemTypes

st.header("Representation of Polymers")

st.markdown(
    """
    In this section, you can build the representation of your polymers. This representation will be the input for the ML models you will train. The representation is built using the SMILES strings of the monomers, which are the building blocks of the polymers.
    We currently support two types of representations:
    1. **Graph Representation**: This representation is built using the SMILES strings of the monomers. The graph representation is built using the RDKit library, which is a collection of cheminformatics and machine learning tools. For this representation, the SMILES strings are converted into a graph representation, where the atoms are the nodes and the bonds are the edges. This representation is used to build the graph neural networks (GNNs) that will be trained on your dataset.
    2. **Molecular Descriptors**: This representation is built using the RDKit library. The molecular descriptors are a set of numerical values that describe the structure and properties of the molecule. This approach effectively transforms the molecule into a vector representation. You can also use descriptors from the dataset, which you can concatenate with the RDkit descriptors or use them as a separate input to the model.
    """
)


choices = get_experiments()
experiment_name = experiment_selector(choices)


if experiment_name:

    experiment_path = polynet_experiments_base_dir() / experiment_name

    path_to_data_opts = data_options_path(
        experiment_path=polynet_experiments_base_dir() / experiment_name
    )

    data_options = load_options(path=path_to_data_opts, options_class=DataOptions)

    path_to_representation_opts = representation_options_path(
        experiment_path=polynet_experiments_base_dir() / experiment_name
    )
    representation_options = load_options(
        path=path_to_representation_opts, options_class=RepresentationOptions
    )

    path_to_train_gnn_options = train_gnn_model_options_path(
        experiment_path=polynet_experiments_base_dir() / experiment_name
    )

    train_gnn_options = load_options(path=path_to_train_gnn_options, options_class=TrainGNNOptions)

    path_to_general_opts = general_options_path(experiment_path=experiment_path)

    general_experiment_options = load_options(
        path=path_to_general_opts, options_class=GeneralConfigOptions
    )

    if not path_to_train_gnn_options.exists() or not path_to_general_opts.exists():
        st.error(
            "No models have been trained yet. Please train a model first in the 'Train GNN' section."
        )
        st.stop()

    display_model_results(experiment_path=experiment_path, expanded=False)

    predictions_path = predictions_path = ml_results_file_path(experiment_path=experiment_path)
    predictions = pd.read_csv(predictions_path, index_col=0)
    metrics_path = model_metrics_file_path(experiment_path=experiment_path)
    with open(metrics_path, "rb") as f:
        metrics = json.load(f)

    if st.checkbox("Show predictions data"):
        st.dataframe(predictions)

    if data_options.problem_type == ProblemTypes.Regression:

        st.subheader("Parity Plot")

        parity_plot = parity_plot_form(
            predictions_df=predictions,
            general_experiment_options=general_experiment_options,
            gnn_training_options=train_gnn_options,
            data_options=data_options,
        )

        if parity_plot:

            st.pyplot(parity_plot, clear_figure=True)

            if st.button("Save Parity Plot"):
                parity_plot_path = experiment_path / "parity_plot.png"
                parity_plot.savefig(parity_plot_path)
                st.success(f"Parity plot saved to {parity_plot_path}")

    elif data_options.problem_type == ProblemTypes.Classification:
        st.subheader("Confusion Matrix")

        confusion_matrix_plot = confusion_matrix_plot_form(
            predictions_df=predictions,
            general_experiment_options=general_experiment_options,
            gnn_training_options=train_gnn_options,
            data_options=data_options,
        )

        if confusion_matrix_plot:
            st.pyplot(confusion_matrix_plot, clear_figure=True)

    st.subheader("Compare models")

    compare_preds_plot = compare_predictions_form(
        predictions_df=predictions,
        target_variable_name=data_options.target_variable_name,
        data_options=data_options,
    )

    if compare_preds_plot:

        st.pyplot(compare_preds_plot)

    st.divider()

    compare_metrics_plot = compare_metrics_form(metrics=metrics, data_options=data_options)

    if compare_metrics_plot:
        st.pyplot(compare_metrics_plot)
