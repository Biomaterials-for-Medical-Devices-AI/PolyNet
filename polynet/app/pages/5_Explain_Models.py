import joblib

import pandas as pd
import streamlit as st

from polynet.app.components.experiments import experiment_selector
from polynet.app.components.forms.explain_model import (
    embedding_projection,
    explain_predictions_form,
)
from polynet.app.components.forms.explain_tml import explain_tml_form
from polynet.app.components.plots import display_model_results, display_unseen_predictions
from polynet.app.options.file_paths import (
    data_options_path,
    general_options_path,
    gnn_raw_data_file,
    gnn_raw_data_path,
    ml_results_file_path,
    model_dir,
    polynet_experiments_base_dir,
    representation_file_path,
    representation_options_path,
    train_gnn_model_options_path,
    train_tml_model_options_path,
)
from polynet.app.services.configurations import load_options
from polynet.app.services.experiments import get_experiments
from polynet.app.services.model_training import load_dataframes
from polynet.config.column_names import get_iterator_name
from polynet.config.schemas import DataConfig, GeneralConfig, RepresentationConfig, SplitConfig
from polynet.featurizer.polymer_graph import CustomPolymerGraph

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
    path_to_data_opts = data_options_path(experiment_path=experiment_path)
    data_options = load_options(path=path_to_data_opts, options_class=DataConfig)

    # load the representation options
    path_to_representation_opts = representation_options_path(experiment_path=experiment_path)
    if not path_to_representation_opts.exists():
        st.error(
            "No representation options were found for this experiment. Please first create a representation, train models, and then come back to this page."
        )
        st.stop()
    representation_options = load_options(
        path=path_to_representation_opts, options_class=RepresentationConfig
    )

    # load gnn options
    path_to_train_gnn_options = train_gnn_model_options_path(experiment_path=experiment_path)
    gnn_exists = path_to_train_gnn_options.exists()

    path_to_train_tml_options = train_tml_model_options_path(experiment_path=experiment_path)
    tml_exists = path_to_train_tml_options.exists()

    if not gnn_exists and not tml_exists:
        st.error(
            "No TML nor GNN options were found. To run explanations, please first train a model."
        )
        st.stop()

    # load general training options
    path_to_general_opts = general_options_path(experiment_path=experiment_path)
    general_experiment_options = load_options(
        path=path_to_general_opts, options_class=GeneralConfig
    )

    path_to_split_options = experiment_path / "split_options.json"
    split_options = load_options(path=path_to_split_options, options_class=SplitConfig)

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
        node_feats=representation_options.node_features,
        edge_feats=representation_options.edge_features,
        polymer_descriptors=representation_options.polymer_descriptors,
    )

    # load the original data
    data = pd.read_csv(
        gnn_raw_data_file(file_name=data_options.data_name, experiment_path=experiment_path),
        index_col=0,
    )

    # load the predictions of the model
    preds = pd.read_csv(ml_results_file_path(experiment_path=experiment_path), index_col=0)

    # get the name of the iterator used for training
    iterator_col = get_iterator_name(split_options.split_type)

    # get the list of trained GNN models
    gnn_models_dir = model_dir(experiment_path=experiment_path)
    gnn_models = [
        model.name
        for model in gnn_models_dir.iterdir()
        if model.is_file() and model.suffix == ".pt"
    ]

    # st.subheader("Graph Embeddings Projection Plot")

    # st.markdown(
    #     """
    # The projection plot shows the graph embeddings of the polymers in a 2D space.
    # You can select a GNN model to get the embeddings from, and to colour the points based on different options.
    # The embeddings are generated by the GNN model and can be used to visualise how the model is organising the latent space.
    # """
    # )

    # embedding_projection(
    #     gnn_models=gnn_models,
    #     gnn_models_dir=gnn_models_dir,
    #     iterator_col=iterator_col,
    #     data=data,
    #     preds=preds,
    #     dataset=dataset,
    #     data_options=data_options,
    #     random_seed=general_experiment_options.random_seed,
    #     weights_col=representation_options.weights_col,
    #     merging_approach=representation_options.smiles_merge_approach,
    # )

    st.header("Model Predictions Explainabilty")

    st.markdown(
        """
        Explain the predictions of your ensemble TML or GNN models.

    Two complementary explanation views are available as separate tabs:

    - **Global Explanation — Population Trends** answers *"which molecular fragments or descriptors consistently drive predictions across a set of molecules?"*
      Results are shown as a ridge plot of fragment attribution distributions, one row per fragment, preserving the full spread across ensemble models.

    - **Local Explanation — Single Molecule** answers *"how does this specific polymer's structure influence its prediction?"*
      Each selected molecule gets its own panel with a per-atom attribution heatmap and a fragment importance table.

    Select your models and settings above the tabs, then run each pipeline independently.
    """
    )

    # ------------------------------------------------------------------
    # TML SHAP Explainability
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Explain TML Model Predictions")

    if path_to_train_tml_options.exists():
        descriptor_dir = representation_file_path(experiment_path=experiment_path)
        tml_models_dir = model_dir(experiment_path=experiment_path)

        tml_model_files = (
            [f for f in tml_models_dir.iterdir() if f.is_file() and f.suffix == ".joblib"]
            if tml_models_dir.exists()
            else []
        )

        descriptor_csvs = (
            [f for f in descriptor_dir.iterdir() if f.is_file() and f.suffix == ".csv"]
            if descriptor_dir.exists()
            else []
        )

        if not tml_model_files:
            st.info(
                "No trained TML models found for this experiment. "
                "Train TML models first to enable SHAP explanations."
            )
        elif not descriptor_csvs:
            st.info(
                "No descriptor CSVs found for this experiment. "
                "Compute molecular descriptors first to enable TML SHAP explanations."
            )
        else:

            # Load TML models from .joblib files
            tml_trained: dict = {}
            for model_file in tml_model_files:
                try:
                    tml_trained[model_file.stem] = joblib.load(model_file)
                except Exception as e:
                    st.warning(f"Could not load TML model `{model_file.name}`: {e}")

            descriptor_dfs = load_dataframes(
                representation_options=representation_options,
                data_options=data_options,
                experiment_path=experiment_path,
            )

            if tml_trained and descriptor_dfs:
                explain_tml_form(
                    experiment_path=experiment_path,
                    tml_models=tml_trained,
                    descriptor_dfs=descriptor_dfs,
                    data_options=data_options,
                    preds=preds,
                )
    else:
        st.error(
            "No TML training options were found. Please first run model training before running TML explanations."
        )

    st.subheader("Explain GNN Model Predictions")

    if path_to_train_gnn_options.exists():
        explain_predictions_form(
            experiment_path=experiment_path,
            gnn_models=gnn_models,
            gnn_models_dir=gnn_models_dir,
            iterator_col=iterator_col,
            data_options=data_options,
            data=data,
            preds=preds,
            dataset=dataset,
        )
    else:
        st.error(
            "No GNN training options were found. Please first run model training before running GNN explanations."
        )
