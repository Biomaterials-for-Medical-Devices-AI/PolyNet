import pandas as pd
import streamlit as st

from polynet.app.components.experiments import experiment_selector
from polynet.app.components.forms.explain_model import (
    embedding_projection,
    explain_predictions_form,
)
from polynet.app.components.plots import display_model_results, display_unseen_predictions
from polynet.app.options.data import DataOptions
from polynet.app.options.file_paths import (
    data_options_path,
    general_options_path,
    gnn_raw_data_file,
    gnn_raw_data_path,
    ml_results_file_path,
    model_dir,
    polynet_experiments_base_dir,
    representation_options_path,
    train_gnn_model_options_path,
)
from polynet.app.options.general_experiment import GeneralConfigOptions
from polynet.app.options.representation import RepresentationOptions
from polynet.app.services.configurations import load_options
from polynet.app.services.experiments import get_experiments
from polynet.featurizer.graph_representation.polymer import CustomPolymerGraph
from polynet.options.col_names import get_iterator_name

st.header("Explain your models")
st.markdown(
    """
In this section, you can explain the predictions of the GNN models. You can select the model you want to explain, the set of data you want to explain, and the specific datapoints you want to explain.

We different explanation levels, including explaining general trends in the model, explanation of the molecular embedding, and the explanation of specific datapoints.

To explain the models or specific instances, you can select the explainability algorithm you want to use, the node features you want to explain, and the colors for the positive and negative explanations. The explanations will be displayed as plots.

For the molecular embedding, you can choose what method of dimensionality reduction to use to get a 2D projection from them. Further, you can select from different options to colour the projection plot, giving insighits about how the model is organising the latent space.
"""
)


choices = get_experiments()
experiment_name = experiment_selector(choices)


if experiment_name:

    # get experiment name
    experiment_path = polynet_experiments_base_dir() / experiment_name

    # load the data options
    path_to_data_opts = data_options_path(
        experiment_path=polynet_experiments_base_dir() / experiment_name
    )
    data_options = load_options(path=path_to_data_opts, options_class=DataOptions)

    # load the representation options
    path_to_representation_opts = representation_options_path(
        experiment_path=polynet_experiments_base_dir() / experiment_name
    )
    if not path_to_representation_opts.exists():
        st.error(
            "No representation options were found for this experiment. Please first create a representation, train models, and then come back to this page."
        )
        st.stop()
    representation_options = load_options(
        path=path_to_representation_opts, options_class=RepresentationOptions
    )

    # load gnn options
    path_to_train_gnn_options = train_gnn_model_options_path(
        experiment_path=polynet_experiments_base_dir() / experiment_name
    )
    if not path_to_train_gnn_options.exists():
        st.error(
            "No GNN training options were found. Please first run model training before running explanations."
        )
        st.stop()

    # load general training options
    path_to_general_opts = general_options_path(experiment_path=experiment_path)
    general_experiment_options = load_options(
        path=path_to_general_opts, options_class=GeneralConfigOptions
    )

    # display the modelling results
    display_model_results(experiment_path=experiment_path, expanded=False)
    display_unseen_predictions(experiment_path=experiment_path)

    # load the graph dataset to run the explanations
    dataset = CustomPolymerGraph(
        filename=data_options.data_name,
        root=gnn_raw_data_path(experiment_path=experiment_path).parent,
        smiles_cols=data_options.smiles_cols,
        target_col=data_options.target_variable_col,
        id_col=data_options.id_col,
        weights_col=representation_options.weights_col,
        node_feats=representation_options.node_feats,
        edge_feats=representation_options.edge_feats,
    )

    # load the original data
    data = pd.read_csv(
        gnn_raw_data_file(file_name=data_options.data_name, experiment_path=experiment_path),
        index_col=0,
    )

    # load the predictions of the model
    preds = pd.read_csv(ml_results_file_path(experiment_path=experiment_path), index_col=0)

    # get the name of the iterator used for training
    iterator_col = get_iterator_name(general_experiment_options.split_type)

    # get the list of trained GNN models
    gnn_models_dir = model_dir(experiment_path=experiment_path)
    gnn_models = [
        model.name
        for model in gnn_models_dir.iterdir()
        if model.is_file() and model.suffix == ".pt"
    ]

    st.subheader("Graph Embeddings Projection Plot")

    st.markdown(
        """
    The projection plot shows the graph embeddings of the polymers in a 2D space.
    You can select a GNN model to get the embeddings from, and to colour the points based on different options.
    The embeddings are generated by the GNN model and can be used to visualise how the model is organising the latent space.
    """
    )

    embedding_projection(
        gnn_models=gnn_models,
        gnn_models_dir=gnn_models_dir,
        iterator_col=iterator_col,
        data=data,
        preds=preds,
        dataset=dataset,
        data_options=data_options,
        random_seed=general_experiment_options.random_seed,
        weights_col=representation_options.weights_col,
    )

    st.subheader("Explain GNN Model Predictions")
    st.markdown(
        """
    In this section, you can explain the predictions of the GNN models. You can select the model you want to explain, the set of data you want to explain, and the specific polymers you want to explain. You can also select the explainability algorithm you want to use, the node features you want to explain, and the colors for the positive and negative explanations. The explanations will be displayed as plots, and you can also plot the t-SNE of the graph embeddings if you wish.
    """
    )
    explain_predictions_form(
        experiment_path=experiment_path,
        gnn_models=gnn_models,
        gnn_models_dir=gnn_models_dir,
        iterator_col=iterator_col,
        data_options=data_options,
        data=data,
        preds=preds,
        dataset=dataset,
        node_feats=representation_options.node_feats,
    )
