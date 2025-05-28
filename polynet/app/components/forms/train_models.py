import streamlit as st

from polynet.app.options.state_keys import (
    GeneralConfigStateKeys,
    TrainGNNStateKeys,
    TrainTMLStateKeys,
)
from polynet.options.enums import NetworkParams, Networks, Pooling, ProblemTypes, SplitMethods


def train_TML_models():

    if st.checkbox("Train traditional machine learning models", value=True, key=TrainTMLStateKeys):
        pass


def train_GNN_models_form():
    if st.checkbox(
        "Train Graph Neural Networks (GNNs)", value=True, key=TrainGNNStateKeys.TrainGNN
    ):
        st.write(
            "Graph Neural Networks (GNNs) are a type of neural network that operates on graph-structured data. They are particularly well-suited for tasks involving molecular structures, such as predicting properties of polymers based on their chemical structure."
        )

        gnn_conv_params = {}

        conv_layers = st.selectbox(
            "Select a GNN convolutional layer to train",
            options=[Networks.GCN, Networks.TransformerGNN],
            # default=[Networks.GCN],
            index=0,
            key=TrainGNNStateKeys.GNNConvolutionalLayers,
        )

        if not conv_layers:
            st.error("Please select at least one GNN convolutional layer to train.")
            st.stop()

        if Networks.GCN in st.session_state[TrainGNNStateKeys.GNNConvolutionalLayers]:
            st.write("#### GCN Hyperparameters")

            st.warning("Beaware that GCN does not support edge features. ")
            # Add GCN specific hyperparameters here
            improved = st.selectbox(
                "Fit bias", options=[True, False], key=TrainGNNStateKeys.Improved, index=0
            )
            gnn_conv_params[Networks.GCN] = {NetworkParams.Improved: improved}

        if Networks.TransformerGNN in st.session_state[TrainGNNStateKeys.GNNConvolutionalLayers]:
            st.write("#### Transformer GNN Hyperparameters")
            # Add Transformer GNN specific hyperparameters here
            num_heads = st.slider(
                "Select the number of attention heads",
                min_value=1,
                max_value=8,
                value=4,
                key=TrainGNNStateKeys.NumHeads,
            )
            gnn_conv_params[Networks.TransformerGNN] = {NetworkParams.NumHeads: num_heads}

        st.write(" ### General GNN Hyperparameters")

        st.select_slider(
            "Select the number of convolutional GNN layers",
            options=list(range(1, 6)),
            value=2,
            key=TrainGNNStateKeys.GNNNumberOfLayers,
        )

        st.select_slider(
            "Select the embedding dimension",
            options=list(range(16, 257, 16)),
            value=128,
            key=TrainGNNStateKeys.GNNEmbeddingDimension,
        )

        st.selectbox(
            "Select the pooling method",
            options=[Pooling.GlobalAddPool, Pooling.GlobalMeanPool, Pooling.GlobalMaxPool],
            key=TrainGNNStateKeys.GNNPoolingMethod,
            index=2,
        )

        st.slider(
            "Select the number of readout layers",
            min_value=1,
            max_value=5,
            value=2,
            key=TrainGNNStateKeys.GNNReadoutLayers,
        )

        st.slider(
            "Select the dropout rate",
            min_value=0.0,
            max_value=0.5,
            value=0.01,
            step=0.01,
            key=TrainGNNStateKeys.GNNDropoutRate,
        )

        st.slider(
            "Select the learning rate",
            min_value=0.0001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            key=TrainGNNStateKeys.GNNLearningRate,
        )

        st.slider(
            "Select the batch size",
            min_value=16,
            max_value=128,
            value=32,
            step=16,
            key=TrainGNNStateKeys.GNNBatchSize,
        )

        return gnn_conv_params


def split_data_form(problem_type: ProblemTypes):

    if problem_type == ProblemTypes.Classification:

        st.selectbox(
            "Select a method to split the data",
            options=[SplitMethods.Random, SplitMethods.Stratified],
            index=1,
            key=GeneralConfigStateKeys.SplitMethod,
        )

        if st.toggle("Balance classes on training set", key=GeneralConfigStateKeys.BalanceClasses):

            st.select_slider(
                "Select the target proportion of classes",
                options=[0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
                value=0.5,
                key=GeneralConfigStateKeys.DesiredProportion,
            )

    else:
        st.selectbox(
            "Select a method to split the data",
            options=[SplitMethods.Random],
            index=0,
            disabled=True,
            key=GeneralConfigStateKeys.SplitMethod,
        )

    st.slider(
        "Select the test split ratio",
        min_value=0.1,
        max_value=0.9,
        value=0.2,
        key=GeneralConfigStateKeys.TestSize,
    )

    st.slider(
        "Select the validation split ratio",
        min_value=0.0,
        max_value=0.9,
        value=0.2,
        key=GeneralConfigStateKeys.ValidationSize,
    )

    st.number_input(
        "Enter a random seed for reproducibility",
        min_value=0,
        max_value=1000000,
        value=1,
        key=GeneralConfigStateKeys.RandomSeed,
    )
    pass
