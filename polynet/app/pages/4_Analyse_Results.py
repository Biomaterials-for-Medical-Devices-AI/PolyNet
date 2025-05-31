import streamlit as st
import pandas as pd
from polynet.app.components.experiments import experiment_selector
from polynet.app.options.file_paths import (
    polynet_experiments_base_dir,
    data_options_path,
    representation_options_path,
    train_gnn_model_options_path,
    ml_gnn_results_file_path,
    general_options_path,
    gnn_raw_data_path,
    gnn_raw_data_file,
)
from polynet.app.options.data import DataOptions
from polynet.app.options.representation import RepresentationOptions
from polynet.app.options.general_experiment import GeneralConfigOptions
from polynet.app.options.train_GNN import TrainGNNOptions

from polynet.app.services.experiments import get_experiments
from polynet.app.services.configurations import load_options
from polynet.app.components.plots import display_model_results

from polynet.options.enums import ProblemTypes

from polynet.app.components.forms.analyse_results import (
    parity_plot_form,
    confusion_matrix_plot_form,
)

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

    predictions_path = predictions_path = ml_gnn_results_file_path(
        experiment_path=experiment_path, file_name="predictions.csv"
    )
    predictions = pd.read_csv(
        ml_gnn_results_file_path(experiment_path=experiment_path, file_name="predictions.csv"),
        index_col=0,
    )

    if data_options.problem_type == ProblemTypes.Regression:

        st.subheader("Parity Plot")

        parity_plot = parity_plot_form(
            predictions_df=predictions,
            general_experiment_options=general_experiment_options,
            gnn_training_options=train_gnn_options,
            data_options=data_options,
        )

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
        st.pyplot(confusion_matrix_plot, clear_figure=True)
