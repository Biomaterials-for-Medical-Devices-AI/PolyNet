import streamlit as st

from polynet.app.options.representation import RepresentationOptions
from polynet.app.options.state_keys import (
    GeneralConfigStateKeys,
    TrainGNNStateKeys,
    TrainTMLStateKeys,
)
from polynet.options.enums import (
    ApplyWeightingToGraph,
    NetworkParams,
    Networks,
    Pooling,
    ProblemTypes,
    SplitMethods,
    SplitTypes,
    TransformDescriptors,
    TradtionalMLModels,
)


def train_TML_models(problem_type: ProblemTypes):

    models = {}

    st.write(
        "Molecular descriptors are numerical representations of molecular structures. These will be used to train traditional machine learning models for the predictive task."
    )

    hyperparameter_tunning = st.toggle(
        "Perform hyperparameter tuning",
        key=TrainTMLStateKeys.PerformHyperparameterTuning,
        help="If enabled, the hyperparameters of the models will be tuned using a grid search. This may take a long time depending on the number of models and hyperparameters selected.",
    )

    st.markdown(
        """
        ### Select the machine learning algorithms you want to train
        """
    )

    st.divider()

    if problem_type == ProblemTypes.Regression:

        if st.toggle("Linear Regression", value=True, key=TrainTMLStateKeys.TrainLinearRegression):

            models[TradtionalMLModels.LinearRegression] = {}

            if not hyperparameter_tunning:
                intercept = st.checkbox(
                    "Fit intercept",
                    value=True,
                    key=TrainTMLStateKeys.LinearRegressionFitIntercept,
                    help="If enabled, the model will fit an intercept term. If disabled, the model will not fit an intercept term.",
                )

                models[TradtionalMLModels.LinearRegression] = {"fit_intercept": intercept}

    elif problem_type == ProblemTypes.Classification:

        if st.toggle(
            "Logistic Regression", value=True, key=TrainTMLStateKeys.TrainLogisticRegression
        ):

            models[TradtionalMLModels.LogisticRegression] = {}

            if not hyperparameter_tunning:

                intercept = st.checkbox(
                    "Fit intercept",
                    value=True,
                    key=TrainTMLStateKeys.LinearRegressionFitIntercept,
                    help="If enabled, the model will fit an intercept term. If disabled, the model will not fit an intercept term.",
                )

    st.divider()

    if st.toggle("Random Forest", value=True, key=TrainTMLStateKeys.TrainRandomForest):
        models[TradtionalMLModels.RandomForest] = {}

        if not hyperparameter_tunning:

            n_estimators = st.slider(
                "Select the number of trees in the forest",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                key=TrainTMLStateKeys.RFNumberEstimators,
            )
            models[TradtionalMLModels.RandomForest]["n_estimators"] = n_estimators

            min_samples_split = st.slider(
                "Select the minimum number of samples required to split an internal node",
                min_value=2,
                max_value=20,
                value=2,
                step=1,
                key=TrainTMLStateKeys.RFMinSamplesSplit,
            )
            models[TradtionalMLModels.RandomForest]["min_samples_split"] = min_samples_split

            min_samples_leaf = st.slider(
                "Select the minimum number of samples required to be at a leaf node",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
                key=TrainTMLStateKeys.RFMinSamplesLeaf,
            )
            models[TradtionalMLModels.RandomForest]["min_samples_leaf"] = min_samples_leaf

            if st.checkbox("Set max depth"):
                max_depth = st.slider(
                    "Select the maximum depth of the tree",
                    min_value=1,
                    max_value=20,
                    value=0,
                    step=1,
                    key=TrainTMLStateKeys.RFMaxDepth,
                )
            else:
                max_depth = None
            models[TradtionalMLModels.RandomForest]["max_depth"] = max_depth

    st.divider()

    if st.toggle(
        "Support Vector Machine", value=True, key=TrainTMLStateKeys.TrainSupportVectorMachine
    ):
        models[TradtionalMLModels.SupportVectorMachine] = {}

        if not hyperparameter_tunning:

            kernel = st.selectbox(
                "Select the kernel type",
                options=["linear", "poly", "rbf", "sigmoid"],
                index=2,
                key=TrainTMLStateKeys.SVMKernel,
            )
            models[TradtionalMLModels.SupportVectorMachine]["kernel"] = kernel

            degree = st.slider(
                "Select the degree of the polynomial kernel function",
                min_value=2,
                max_value=5,
                value=3,
                step=1,
                key=TrainTMLStateKeys.SVMDegree,
            )
            models[TradtionalMLModels.SupportVectorMachine]["degree"] = degree

            c = st.slider(
                "Select the regularization parameter C",
                min_value=0.01,
                max_value=10.0,
                value=1.0,
                step=0.01,
                key=TrainTMLStateKeys.SVMC,
            )
            models[TradtionalMLModels.SupportVectorMachine]["C"] = c

    st.divider()

    if st.toggle("XGBoost", value=True, key=TrainTMLStateKeys.TrainXGBoost):
        models[TradtionalMLModels.XGBoost] = {}

        if not hyperparameter_tunning:

            num_estimators = st.slider(
                "Select the number of trees in the forest",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                key=TrainTMLStateKeys.XGBNumberEstimators,
            )
            models[TradtionalMLModels.XGBoost]["n_estimators"] = num_estimators

            learning_rate = st.slider(
                "Select the learning rate",
                min_value=0.001,
                max_value=0.1,
                value=0.01,
                step=0.001,
                key=TrainTMLStateKeys.XGBLearningRate,
            )
            models[TradtionalMLModels.XGBoost]["learning_rate"] = learning_rate

            subsample_size = st.slider(
                "Select the subsample size",
                min_value=0.1,
                max_value=1.0,
                value=0.8,
                step=0.01,
                key=TrainTMLStateKeys.XGBSubsampleSize,
            )
            models[TradtionalMLModels.XGBoost]["subsample"] = subsample_size

            max_depth = st.slider(
                "Select the maximum depth of the tree",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
                key=TrainTMLStateKeys.XGBMaxDepth,
            )
            models[TradtionalMLModels.XGBoost]["max_depth"] = max_depth

    st.divider()

    st.selectbox(
        "Apply transformation to the independent variables",
        options=[
            TransformDescriptors.NoTransformation,
            TransformDescriptors.StandardScaler,
            TransformDescriptors.MinMaxScaler,
        ],
        index=0,
        key=TrainTMLStateKeys.TrasformFeatures,
    )

    return models


def train_GNN_models_form(representation_opts: RepresentationOptions, problem_type: ProblemTypes):

    st.write(
        "Graph Neural Networks (GNNs) are a type of neural network that operates on graph-structured data. They are particularly well-suited for tasks involving molecular structures, such as predicting properties of polymers based on their chemical structure."
    )

    gnn_conv_params = {}

    conv_layers = st.multiselect(
        "Select a GNN convolutional layer to train",
        options=[
            Networks.GCN,
            Networks.GraphSAGE,
            Networks.TransformerGNN,
            Networks.GAT,
            Networks.MPNN,
            Networks.CGGNN,
        ],
        default=[Networks.GCN],
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

    if Networks.GraphSAGE in st.session_state[TrainGNNStateKeys.GNNConvolutionalLayers]:
        st.write("#### GraphSAGE Hyperparameters")

        st.warning("Beaware that GCN does not support edge features. ")
        # Add GraphSAGE specific hyperparameters here
        bias = st.selectbox("Fit bias", options=[True, False], key=TrainGNNStateKeys.Bias, index=0)
        gnn_conv_params[Networks.GraphSAGE] = {NetworkParams.Bias: bias}

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

    if Networks.GAT in st.session_state[TrainGNNStateKeys.GNNConvolutionalLayers]:
        st.write("#### GAT Hyperparameters")
        # Add GAT specific hyperparameters here
        num_heads = st.slider(
            "Select the number of attention heads",
            min_value=1,
            max_value=8,
            value=4,
            key=TrainGNNStateKeys.NHeads,
        )
        gnn_conv_params[Networks.GAT] = {NetworkParams.NumHeads: num_heads}

    if Networks.MPNN in st.session_state[TrainGNNStateKeys.GNNConvolutionalLayers]:
        st.write("#### MPNN Hyperparameters")
        # Add MPNN specific hyperparameters here

        gnn_conv_params[Networks.MPNN] = {}

    if Networks.CGGNN in st.session_state[TrainGNNStateKeys.GNNConvolutionalLayers]:
        st.write("#### CGCGNN Hyperparameters")
        # Add MPNN specific hyperparameters here

        gnn_conv_params[Networks.CGGNN] = {}

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

    if representation_opts.weights_col:
        st.selectbox(
            "Select when you would like to apply the weighting to the graph",
            options=[ApplyWeightingToGraph.BeforeMPP, ApplyWeightingToGraph.BeforePooling],
            index=0,
            key=TrainGNNStateKeys.GNNMonomerWeighting,
        )
    else:
        st.session_state[TrainGNNStateKeys.GNNMonomerWeighting] = ApplyWeightingToGraph.NoWeighting

    if problem_type == ProblemTypes.Classification:
        if st.checkbox(
            "Apply asymmetric loss function", value=True, key=TrainGNNStateKeys.AsymmetricLoss
        ):
            st.slider(
                "Set the imbalance strength",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                key=TrainGNNStateKeys.ImbalanceStrength,
                help="Controls how much strength to apply to the asymmetric loss function. A value of 1 means that weighting will be given by the inverse of the total count of each label, while a value of 0 means that each class will be weighted equally.",
            )

    return gnn_conv_params


def split_data_form(problem_type: ProblemTypes):

    split_type = st.selectbox(
        "Select the split method",
        options=[SplitTypes.TrainValTest],
        index=0,
        disabled=True,
        key=GeneralConfigStateKeys.SplitType,
    )

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

    if split_type == SplitTypes.TrainValTest:
        st.select_slider(
            "Select the number of bootstrap iterations",
            options=list(range(1, 11)),
            value=1,
            key=GeneralConfigStateKeys.BootstrapIterations,
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
        max_value=100000000,
        value=1221,
        key=GeneralConfigStateKeys.RandomSeed,
    )
