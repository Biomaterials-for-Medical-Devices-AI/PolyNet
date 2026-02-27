import streamlit as st

from polynet.app.options.state_keys import (
    GeneralConfigStateKeys,
    TrainGNNStateKeys,
    TrainTMLStateKeys,
)
from polynet.config.enums import (
    ApplyWeightingToGraph,
    ArchitectureParam,
    Network,
    Pooling,
    ProblemType,
    SplitMethod,
    SplitType,
    TraditionalMLModel,
    TrainingParam,
    TransformDescriptor,
)
from polynet.config.schemas.representation import RepresentationConfig


def train_TML_models(problem_type: ProblemType):

    models = {}

    st.write(
        "Molecular descriptors are numerical representations of molecular structures. These will be used to train traditional machine learning models for the predictive task."
    )

    if st.toggle("Train TML models", key=TrainTMLStateKeys.TrainTML):

        hyperparameter_tunning = st.checkbox(
            "Perform hyperparameter tuning",
            key=TrainTMLStateKeys.PerformHyperparameterTuning,
            help="If enabled, the hyperparameters of the models will be tuned using a grid search. This may take a long time depending on the number of models and hyperparameters selected.",
        )

        st.markdown(
            """
            ### Select the machine learning algorithms you want to train
            """
        )

        if problem_type == ProblemType.Regression:

            if st.toggle(
                "Linear Regression", value=False, key=TrainTMLStateKeys.TrainLinearRegression
            ):

                models[TraditionalMLModel.LinearRegression] = {}

                if not hyperparameter_tunning:
                    intercept = st.checkbox(
                        "Fit intercept",
                        value=True,
                        key=TrainTMLStateKeys.LinearRegressionFitIntercept,
                        help="If enabled, the model will fit an intercept term. If disabled, the model will not fit an intercept term.",
                    )

                    models[TraditionalMLModel.LinearRegression] = {"fit_intercept": intercept}

        elif problem_type == ProblemType.Classification:

            if st.toggle(
                "Logistic Regression", value=False, key=TrainTMLStateKeys.TrainLogisticRegression
            ):

                models[TraditionalMLModel.LogisticRegression] = {}

                if not hyperparameter_tunning:

                    penalty = st.selectbox(
                        "Select the norm of the penalty",
                        ["l1", "l2"],
                        index=0,
                        key=TrainTMLStateKeys.LogisticRegressionPenalty,
                    )

                    C = st.selectbox(
                        "Select the inverse of regularization strength",
                        [0.1, 1, 10, 100],
                        index=1,
                        key=TrainTMLStateKeys.LogisticRegressionC,
                    )

                    intercept = st.checkbox(
                        "Fit intercept",
                        value=True,
                        key=TrainTMLStateKeys.LinearRegressionFitIntercept,
                        help="If enabled, the model will fit an intercept term. If disabled, the model will not fit an intercept term.",
                    )

                    solver = st.selectbox(
                        "Select the solver",
                        ["lbfgs", "liblinear"],
                        index=1,
                        key=TrainTMLStateKeys.LogisticRegressionSolver,
                    )

                    models[TraditionalMLModel.LogisticRegression] = {
                        "penalty": penalty,
                        "C": C,
                        "fit_intercept": intercept,
                        "solver": solver,
                    }

        st.divider()

        if st.toggle("Random Forest", value=False, key=TrainTMLStateKeys.TrainRandomForest):
            models[TraditionalMLModel.RandomForest] = {}

            if not hyperparameter_tunning:

                n_estimators = st.slider(
                    "Select the number of trees in the forest",
                    min_value=10,
                    max_value=1000,
                    value=100,
                    step=10,
                    key=TrainTMLStateKeys.RFNumberEstimators,
                )
                models[TraditionalMLModel.RandomForest]["n_estimators"] = n_estimators

                min_samples_split = st.slider(
                    "Select the minimum number of samples required to split an internal node",
                    min_value=2,
                    max_value=20,
                    value=2,
                    step=1,
                    key=TrainTMLStateKeys.RFMinSamplesSplit,
                )
                models[TraditionalMLModel.RandomForest]["min_samples_split"] = min_samples_split

                min_samples_leaf = st.slider(
                    "Select the minimum number of samples required to be at a leaf node",
                    min_value=1,
                    max_value=20,
                    value=1,
                    step=1,
                    key=TrainTMLStateKeys.RFMinSamplesLeaf,
                )
                models[TraditionalMLModel.RandomForest]["min_samples_leaf"] = min_samples_leaf

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
                models[TraditionalMLModel.RandomForest]["max_depth"] = max_depth

        st.divider()

        if st.toggle(
            "Support Vector Machine", value=False, key=TrainTMLStateKeys.TrainSupportVectorMachine
        ):
            models[TraditionalMLModel.SupportVectorMachine] = {}

            if not hyperparameter_tunning:

                kernel = st.selectbox(
                    "Select the kernel type",
                    options=["linear", "poly", "rbf", "sigmoid"],
                    index=2,
                    key=TrainTMLStateKeys.SVMKernel,
                )
                models[TraditionalMLModel.SupportVectorMachine]["kernel"] = kernel

                degree = st.slider(
                    "Select the degree of the polynomial kernel function",
                    min_value=2,
                    max_value=5,
                    value=3,
                    step=1,
                    key=TrainTMLStateKeys.SVMDegree,
                )
                models[TraditionalMLModel.SupportVectorMachine]["degree"] = degree

                c = st.slider(
                    "Select the regularization parameter C",
                    min_value=0.01,
                    max_value=10.0,
                    value=1.0,
                    step=0.01,
                    key=TrainTMLStateKeys.SVMC,
                )
                models[TraditionalMLModel.SupportVectorMachine]["C"] = c

                if problem_type == ProblemType.Classification:
                    models[TraditionalMLModel.SupportVectorMachine]["probability"] = True

        st.divider()

        if st.toggle("XGBoost", value=False, key=TrainTMLStateKeys.TrainXGBoost):
            models[TraditionalMLModel.XGBoost] = {}

            if not hyperparameter_tunning:

                num_estimators = st.slider(
                    "Select the number of trees in the forest",
                    min_value=10,
                    max_value=1000,
                    value=100,
                    step=10,
                    key=TrainTMLStateKeys.XGBNumberEstimators,
                )
                models[TraditionalMLModel.XGBoost]["n_estimators"] = num_estimators

                learning_rate = st.slider(
                    "Select the learning rate",
                    min_value=0.001,
                    max_value=0.1,
                    value=0.01,
                    step=0.001,
                    key=TrainTMLStateKeys.XGBLearningRate,
                )
                models[TraditionalMLModel.XGBoost]["learning_rate"] = learning_rate

                subsample_size = st.slider(
                    "Select the subsample size",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.8,
                    step=0.01,
                    key=TrainTMLStateKeys.XGBSubsampleSize,
                )
                models[TraditionalMLModel.XGBoost]["subsample"] = subsample_size

                max_depth = st.slider(
                    "Select the maximum depth of the tree",
                    min_value=0,
                    max_value=20,
                    value=0,
                    step=1,
                    key=TrainTMLStateKeys.XGBMaxDepth,
                )
                models[TraditionalMLModel.XGBoost]["max_depth"] = max_depth

        st.divider()

        st.selectbox(
            "Apply transformation to the independent variables",
            options=[
                TransformDescriptor.NoTransformation,
                TransformDescriptor.StandardScaler,
                TransformDescriptor.MinMaxScaler,
            ],
            index=0,
            key=TrainTMLStateKeys.TrasformFeatures,
        )

    return models


def train_GNN_models_form(representation_opts: RepresentationConfig, problem_type: ProblemType):

    st.write(
        "Graph Neural Networks (GNNs) are a type of neural network that operates on graph-structured data. They are particularly well-suited for tasks involving molecular structures, such as predicting properties of polymers based on their chemical structure."
    )

    gnn_conv_params = {}

    if not st.toggle("Train GNN models", key=TrainGNNStateKeys.TrainGNN):
        return gnn_conv_params

    hyperparameter_tunning = st.checkbox(
        "Perform hyperparameter tuning",
        key=TrainGNNStateKeys.HypTunning,
        help="If enabled, hyperparameters will be tuned via grid search (can be slow).",
    )

    st.markdown("### Select the GNN convolutional layers you want to train")

    conv_layers = st.multiselect(
        "Select GNN convolutional layers to train",
        options=[
            Network.GCN,
            Network.GraphSAGE,
            Network.TransformerGNN,
            Network.GAT,
            Network.MPNN,
            Network.CGGNN,
        ],
        default=[Network.GCN],
        key=TrainGNNStateKeys.GNNConvolutionalLayers,
    )

    if not conv_layers:
        st.error("Please select at least one GNN convolutional layer to train.")
        st.stop()

    if not hyperparameter_tunning:
        share_params = st.checkbox(
            "Use same hyperparameter values for shared GNN parameters across all architectures.",
            value=True,
            key=TrainGNNStateKeys.SharedGNNParams,
            help="If enabled, shared GNN hyperparameters (e.g., layers, embedding dim, pooling) are the same across all architectures.",
        )
    else:
        share_params = None

    # Define per-network UI in a simple, modular way
    def gcn_ui():
        st.warning("Be aware that `GCN` does not support edge features.")
        improved = st.selectbox("Fit bias", [True, False], key=TrainGNNStateKeys.Improved)
        return {ArchitectureParam.Improved: improved}

    def sage_ui():
        st.warning("Be aware that `GraphSAGE` does not support edge features.")
        bias = st.selectbox("Fit bias", [True, False], key=TrainGNNStateKeys.Bias)
        return {ArchitectureParam.Bias: bias}

    def transformer_ui():
        num_heads = st.slider("Number of attention heads", 1, 8, 4, key=TrainGNNStateKeys.NumHeads)
        return {ArchitectureParam.NumHeads: num_heads}

    def gat_ui():
        num_heads = st.slider("Number of attention heads", 1, 8, 4, key=TrainGNNStateKeys.NHeads)
        return {ArchitectureParam.NumHeads: num_heads}

    def empty_ui(msg):
        st.write(msg)
        return {}

    # Central configuration
    NETWORK_UIS = {
        Network.GCN: ("GCN Hyperparameters", gcn_ui),
        Network.GraphSAGE: ("GraphSAGE Hyperparameters", sage_ui),
        Network.TransformerGNN: ("Transformer GNN Hyperparameters", transformer_ui),
        Network.GAT: ("GAT Hyperparameters", gat_ui),
        Network.MPNN: (
            "MPNN Hyperparameters",
            lambda: empty_ui("Currently, no specific parameters for `MPNN` are available."),
        ),
        Network.CGGNN: (
            "CGGNN Hyperparameters",
            lambda: empty_ui("Currently, no specific parameters for `CGGNN` are available."),
        ),
    }

    for network in conv_layers:
        title, ui_func = NETWORK_UIS[network]
        if not hyperparameter_tunning:
            st.write(f"#### {title}")
            specific_params = ui_func()
            st.divider()
        else:
            specific_params = {}

        gnn_conv_params[network] = specific_params

        # Handle non-shared parameters if applicable
        if not share_params and not hyperparameter_tunning:
            shared = GNN_shared_params_form(
                representation_opts=representation_opts, problem_type=problem_type, network=network
            )
            st.divider()
            gnn_conv_params[network].update(shared)

    # Handle shared parameters once at the end
    if share_params:
        st.markdown("#### Set shared GNN hyperparameters")
        shared_params = GNN_shared_params_form(
            representation_opts=representation_opts, problem_type=problem_type, network=None
        )
        for net in gnn_conv_params:
            gnn_conv_params[net].update(shared_params)

    return gnn_conv_params


def GNN_shared_params_form(
    representation_opts: RepresentationConfig, problem_type: ProblemType, network: Network = None
):

    shared_params = {}

    with st.expander("General GNN Hyperparameters", expanded=True):

        conv_layers = st.select_slider(
            "Select the number of convolutional GNN layers",
            options=list(range(1, 6)),
            value=2,
            key=TrainGNNStateKeys.GNNNumberOfLayers + network.value if network else "",
        )

        emb_dim = st.select_slider(
            "Select the embedding dimension",
            options=list(range(16, 257, 16)),
            value=128,
            key=TrainGNNStateKeys.GNNEmbeddingDimension + network.value if network else "",
        )

        pooling = st.selectbox(
            "Select the pooling method",
            options=[Pooling.GlobalAddPool, Pooling.GlobalMeanPool, Pooling.GlobalMaxPool],
            key=TrainGNNStateKeys.GNNPoolingMethod + network.value if network else "",
            index=2,
        )

        readout_layers = st.slider(
            "Select the number of readout layers",
            min_value=1,
            max_value=5,
            value=2,
            key=TrainGNNStateKeys.GNNReadoutLayers + network.value if network else "",
        )

        dropout = st.slider(
            "Select the dropout rate",
            min_value=0.0,
            max_value=0.5,
            value=0.01,
            step=0.01,
            key=TrainGNNStateKeys.GNNDropoutRate + network.value if network else "",
        )

        learning_rate = st.slider(
            "Select the learning rate",
            min_value=0.0001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            key=TrainGNNStateKeys.GNNLearningRate + network.value if network else "",
        )

        batch_size = st.slider(
            "Select the batch size",
            min_value=16,
            max_value=128,
            value=32,
            step=16,
            key=TrainGNNStateKeys.GNNBatchSize + network.value if network else "",
        )

        if representation_opts.weights_col:
            apply_weighting = st.selectbox(
                "Select when you would like to apply the weighting to the graph",
                options=[ApplyWeightingToGraph.BeforeMPP, ApplyWeightingToGraph.BeforePooling],
                index=1,
                key=TrainGNNStateKeys.GNNMonomerWeighting + network.value if network else "",
            )
        else:
            apply_weighting = st.session_state[TrainGNNStateKeys.GNNMonomerWeighting] = (
                ApplyWeightingToGraph.NoWeighting
            )

        if problem_type == ProblemType.Classification:
            if st.checkbox(
                "Apply asymmetric loss function",
                value=False,
                key=TrainGNNStateKeys.AsymmetricLoss + network.value if network else "",
            ):
                assym_loss_strength = st.slider(
                    "Set the imbalance strength",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    key=TrainGNNStateKeys.ImbalanceStrength + network.value if network else "",
                    help="Controls how much strength to apply to the asymmetric loss function. A value of 1 means that weighting will be given by the inverse of the total count of each label, while a value of 0 means that each class will be weighted equally.",
                )
            else:
                assym_loss_strength = None

            shared_params[TrainingParam.AsymmetricLossStrength] = assym_loss_strength

    shared_params[ArchitectureParam.NumConvolutions] = conv_layers
    shared_params[ArchitectureParam.EmbeddingDim] = emb_dim
    shared_params[ArchitectureParam.PoolingMethod] = pooling
    shared_params[ArchitectureParam.ReadoutLayers] = readout_layers
    shared_params[ArchitectureParam.Dropout] = dropout
    shared_params[TrainingParam.LearningRate] = learning_rate
    shared_params[TrainingParam.BatchSize] = batch_size
    shared_params[ArchitectureParam.ApplyWeightingGraph] = apply_weighting

    return shared_params


def split_data_form(problem_type: ProblemType):

    split_type = st.selectbox(
        "Select the split method",
        options=[SplitType.TrainValTest],
        index=0,
        disabled=True,
        key=GeneralConfigStateKeys.SplitType,
    )

    if problem_type == ProblemType.Classification:

        st.selectbox(
            "Select a method to split the data",
            options=[SplitMethod.Random, SplitMethod.Stratified],
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
            options=[SplitMethod.Random],
            index=0,
            disabled=True,
            key=GeneralConfigStateKeys.SplitMethod,
        )

    if split_type == SplitType.TrainValTest:
        st.select_slider(
            "Select the number of bootstrap iterations",
            options=list(range(1, 11)),
            value=1,
            key=GeneralConfigStateKeys.BootstrapIterations,
        )

    st.slider(
        "Select the test split ratio",
        min_value=0.01,
        max_value=0.9,
        value=0.2,
        key=GeneralConfigStateKeys.TestSize,
    )

    st.slider(
        "Select the validation split ratio",
        min_value=0.01,
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
