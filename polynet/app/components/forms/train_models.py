import streamlit as st

from polynet.app.options.state_keys import (
    GeneralConfigStateKeys,
    TrainGNNStateKeys,
    TrainTMLStateKeys,
)
from polynet.config.constants import POLYBERT_MODEL
from polynet.config.enums import (
    ApplyWeightingToGraph,
    ArchitectureParam,
    FeatureSelection,
    MolecularDescriptor,
    Network,
    Optimizer,
    Pooling,
    ProblemType,
    RegressionLoss,
    Scheduler,
    SplitMethod,
    SplitSampler,
    SplitType,
    TargetTransformDescriptor,
    TraditionalMLModel,
    TrainingParam,
    TransformDescriptor,
)
from polynet.config.schemas.feature_preprocessing import FeatureTransformConfig
from polynet.config.schemas.fingerprints import MorganFingerprintConfig, RDKitFingerprintConfig
from polynet.config.schemas.representation import RepresentationConfig
from polynet.config.schemas.split_data import (
    DETERMINISTIC_SAMPLERS,
    SAMPLERS_USING_FINGERPRINTS,
    available_samplers,
    deterministic_sampler_warning,
)
from polynet.config.schemas.target_preprocessing import TargetTransformConfig
from polynet.config.schemas.training import GNNOptimisationConfig, TrainGNNConfig


def train_TML_models(problem_type: ProblemType) -> dict:

    models = {}

    st.write(
        "Molecular descriptors are numerical representations of molecular structures. These will be used to train traditional machine learning models for the predictive task."
    )

    if st.toggle("Train TML models", key=TrainTMLStateKeys.TrainTML):

        st.toggle(
            "Include the validation set in TML training",
            value=True,
            key=TrainTMLStateKeys.IncludeValidation,
            help="On (default): TML models, their feature scaling and their hyperparameter "
            "search use the training and validation samples of each split. Off: they use "
            "the training samples only — the same data the GNNs train on — and the "
            "validation samples are scored as a held-out validation set.",
        )

        hyperparameter_tunning = st.checkbox(
            "Perform hyperparameter tuning",
            key=TrainTMLStateKeys.PerformHyperparameterTuning,
            help="If enabled, the hyperparameters of the models will be tuned with a randomised search (30 configurations sampled from a predefined grid, scored by k-fold shuffled cross-validation). This may take a long time depending on the number of models selected.",
        )

        if hyperparameter_tunning:
            st.number_input(
                "Number of cross-validation folds (k)",
                min_value=2,
                value=5,
                step=1,
                key=TrainTMLStateKeys.HPONumFolds,
                help="Folds used to score each configuration (shuffled; stratified for "
                "classification). k must not exceed the number of training samples or, for "
                "classification, the size of the smallest class — this is checked before "
                "training starts.",
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

    return models


def feature_transformer_widgets(
    train_tml: bool, gnn_polymer_descriptors: bool
) -> FeatureTransformConfig:
    """
    Render the pipeline-wide feature preprocessing widgets.

    The scaler applies to every tabular feature the pipeline uses: the
    molecular descriptors of traditional ML models and the user-supplied
    polymer descriptors concatenated to the GNN graph embedding. Feature
    selection is only offered when traditional ML models are trained, since
    it does not apply to GNNs.

    Parameters
    ----------
    train_tml:
        Whether traditional ML models will be trained.
    gnn_polymer_descriptors:
        Whether GNNs will be trained with user-supplied polymer descriptors.

    Returns
    -------
    FeatureTransformConfig
        The selected scaler and (TML only) feature selection steps.
    """
    applies_to = []
    if train_tml:
        applies_to.append("the molecular descriptors of the TML models")
    if gnn_polymer_descriptors:
        applies_to.append("the polymer descriptors concatenated to the GNN graph embedding")
    st.caption(
        "Scaling is fitted on the training set of each split and applied to "
        + " and to ".join(applies_to)
        + "."
    )

    scaler = st.selectbox(
        "Scaling / normalization",
        options=[
            TransformDescriptor.NoTransformation,
            TransformDescriptor.StandardScaler,
            TransformDescriptor.MinMaxScaler,
            TransformDescriptor.RobustScaler,
            TransformDescriptor.PowerTransformer,
            TransformDescriptor.QuantileTransformer,
            TransformDescriptor.Normalizer,
        ],
        index=0,
        key=getattr(TrainTMLStateKeys, "FeatureScaler", "FeatureScaler"),
        help="Applied to X (independent variables). Fit on training set, reused for val/test.",
    )

    # Feature selection applies to TML models only.
    enable_fs = train_tml and st.toggle(
        "Enable feature selection (TML models only)",
        value=False,
        key=getattr(TrainTMLStateKeys, "EnableFeatureSelection", "EnableFeatureSelection"),
        help="Applies selection after scaling. Steps are applied sequentially in the order chosen.",
    )

    selectors: dict[FeatureSelection, dict] = {}

    if enable_fs:
        with st.expander("Feature selection settings", expanded=True):
            # Order matters. We’ll let the user decide order via a multiselect + order list.
            available_steps = [FeatureSelection.Variance, FeatureSelection.Correlation]

            chosen_steps = st.multiselect(
                "Select feature selection steps (order matters)",
                options=available_steps,
                default=[FeatureSelection.Variance],
                key=getattr(TrainTMLStateKeys, "FeatureSelectionSteps", "FeatureSelectionSteps"),
                help="Steps are applied in the order selected below.",
            )

            # Per-step params
            for step in chosen_steps:
                if step == FeatureSelection.Variance:
                    thr = st.number_input(
                        "Variance threshold",
                        min_value=0.0,
                        value=0.0,
                        step=1e-6,
                        format="%.6f",
                        key=getattr(TrainTMLStateKeys, "VarianceThreshold", "VarianceThreshold"),
                        help="Remove features with variance <= threshold (computed after scaling).",
                    )
                    selectors[FeatureSelection.Variance] = {"threshold": float(thr)}

                elif step == FeatureSelection.Correlation:
                    corr_thr = st.slider(
                        "Correlation threshold",
                        min_value=0.50,
                        max_value=0.999,
                        value=0.95,
                        step=0.01,
                        key=getattr(
                            TrainTMLStateKeys, "CorrelationThreshold", "CorrelationThreshold"
                        ),
                        help="Remove one of each pair of features with abs(corr) >= threshold (greedy).",
                    )
                    selectors[FeatureSelection.Correlation] = {"threshold": float(corr_thr)}

            st.caption(
                "Tip: a common choice is Variance → Correlation. "
                "Correlation selection can be expensive for very wide descriptor sets."
            )

    # Return a validated config object (matches your updated schema)
    return FeatureTransformConfig(scaler=scaler, selectors=selectors, random_state=42)


def target_transform_widget() -> TargetTransformConfig:
    """
    Render the target variable scaling widget for regression experiments.

    Returns a :class:`TargetTransformConfig` built from the user's choices.
    The widget is always rendered; callers are responsible for only calling
    this function when the problem type is regression.
    """
    st.markdown("### Target variable scaling")
    st.caption(
        "Optionally scale the target variable before training. "
        "The scaler is fitted on the training set only and inverse-transformed "
        "before metrics and plots are computed, so all reported values remain "
        "in the original target range."
    )

    _STRATEGY_LABELS: dict[TargetTransformDescriptor, str] = {
        TargetTransformDescriptor.NoTransformation: "No scaling",
        TargetTransformDescriptor.StandardScaler: "Standardisation  (zero mean, unit variance)",
        TargetTransformDescriptor.MinMaxScaler: "Min–Max  (scale to [0, 1])",
        TargetTransformDescriptor.RobustScaler: "Robust scaler  (IQR-based, outlier-resistant)",
        TargetTransformDescriptor.Log10: "Log₁₀  (requires all targets > 0)",
        TargetTransformDescriptor.Log1p: "Log(1 + y)  (requires all targets > −1)",
    }

    strategy = st.selectbox(
        "Scaling strategy",
        options=list(_STRATEGY_LABELS.keys()),
        format_func=lambda s: _STRATEGY_LABELS[s],
        index=0,
        key=TrainTMLStateKeys.TargetTransformStrategy,
        help=(
            "Choose how the target variable is scaled before model training. "
            "Log₁₀ and Log(1+y) are useful for targets spanning several orders "
            "of magnitude."
        ),
    )

    if strategy in (TargetTransformDescriptor.Log10, TargetTransformDescriptor.Log1p):
        st.info(
            f"{'Log₁₀' if strategy == TargetTransformDescriptor.Log10 else 'Log(1+y)'} "
            "will be applied. Make sure all training target values satisfy the "
            f"domain constraint ({'> 0' if strategy == TargetTransformDescriptor.Log10 else '> −1'})."
        )

    return TargetTransformConfig(strategy=strategy)


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
        help="If enabled, hyperparameters will be tuned by randomly sampling configurations from a predefined search grid with Ray Tune (can be slow).",
    )

    st.number_input(
        "Number of training epochs",
        min_value=1,
        value=TrainGNNConfig.model_fields["epochs"].default,
        step=10,
        key=TrainGNNStateKeys.Epochs,
        help="Epochs each GNN is trained for (the weights of the epoch with the lowest "
        "validation loss are kept). Hyperparameter-tuning trials train for the same number "
        "of epochs.",
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


def gnn_optimisation_widgets(problem_type: ProblemType) -> GNNOptimisationConfig:
    """
    Render the "Advanced training options" expander for GNNs.

    Lets the user choose the optimiser, the learning-rate scheduler (with only
    the parameters that scheduler uses) and, for regression, the loss. The
    defaults reproduce PolyNet's historical settings (Adam, ReduceLROnPlateau
    with factor 0.9 / patience 15 / min_lr 1e-8, RMSE loss). The same
    settings are used for final training and for every HPO trial.

    Parameters
    ----------
    problem_type:
        Classification or regression (the loss choice is regression-only).

    Returns
    -------
    GNNOptimisationConfig
        The selected settings.
    """
    defaults = GNNOptimisationConfig()
    options: dict = {}

    with st.expander("Advanced training options (optimiser, scheduler, loss)", expanded=False):
        st.caption(
            "Applied to final training and to every hyperparameter-optimisation trial. "
            "The defaults reproduce PolyNet's standard settings."
        )
        options["optimizer"] = st.selectbox(
            "Optimiser",
            options=list(Optimizer),
            index=list(Optimizer).index(defaults.optimizer),
            key=TrainGNNStateKeys.Optimizer,
        )
        scheduler = st.selectbox(
            "Learning-rate scheduler",
            options=list(Scheduler),
            index=list(Scheduler).index(defaults.scheduler),
            key=TrainGNNStateKeys.Scheduler,
            help="reduce_lr_on_plateau lowers the learning rate when the validation loss "
            "stops improving; the other schedulers decay it on a fixed epoch schedule.",
        )
        options["scheduler"] = scheduler
        options["scheduler_factor"] = st.number_input(
            "Decay factor (gamma)",
            min_value=0.01,
            max_value=0.99,
            value=defaults.scheduler_factor,
            step=0.01,
            key=TrainGNNStateKeys.SchedulerFactor,
            help="The learning rate is multiplied by this factor at each decay.",
        )
        if scheduler == Scheduler.ReduceLROnPlateau:
            options["scheduler_patience"] = st.number_input(
                "Patience (epochs)",
                min_value=0,
                value=defaults.scheduler_patience,
                step=1,
                key=TrainGNNStateKeys.SchedulerPatience,
            )
            options["scheduler_min_lr"] = st.number_input(
                "Minimum learning rate",
                min_value=0.0,
                value=defaults.scheduler_min_lr,
                format="%.1e",
                key=TrainGNNStateKeys.SchedulerMinLR,
            )
        elif scheduler == Scheduler.StepLR:
            options["scheduler_step_size"] = st.number_input(
                "Decay every N epochs",
                min_value=1,
                value=defaults.scheduler_step_size,
                step=1,
                key=TrainGNNStateKeys.SchedulerStepSize,
            )
        elif scheduler == Scheduler.MultiStepLR:
            milestones = st.text_input(
                "Decay at epochs (comma-separated)",
                value=", ".join(str(m) for m in defaults.scheduler_milestones),
                key=TrainGNNStateKeys.SchedulerMilestones,
            )
            try:
                options["scheduler_milestones"] = [
                    int(m) for m in milestones.replace(" ", "").split(",") if m
                ]
            except ValueError:
                st.error("Milestones must be whole numbers separated by commas, e.g. 30, 60, 90.")
                st.stop()

        if problem_type == ProblemType.Regression:
            options["regression_loss"] = st.selectbox(
                "Regression loss",
                options=list(RegressionLoss),
                index=list(RegressionLoss).index(defaults.regression_loss),
                key=TrainGNNStateKeys.RegressionLoss,
                help="rmse: root mean squared error (default); mse: mean squared error; "
                "mae: mean absolute error (less sensitive to outliers).",
            )
        else:
            st.caption("Classification models are trained with cross-entropy loss.")

    try:
        return GNNOptimisationConfig(**options)
    except ValueError as e:
        st.error(str(e))
        st.stop()


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
                options=[
                    ApplyWeightingToGraph.BeforeMPP,
                    ApplyWeightingToGraph.BeforePooling,
                    ApplyWeightingToGraph.PerMonomerPooling,
                ],
                index=2,
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


_SAMPLER_HELP = {
    SplitSampler.Random: "Random split.",
    SplitSampler.KennardStone: "Kennard–Stone: training set spans the fingerprint space "
    "(deterministic).",
    SplitSampler.SPXY: "SPXY: like Kennard–Stone on fingerprints and target values "
    "(deterministic).",
    SplitSampler.KMeans: "k-means clusters of fingerprints; each cluster stays in one set.",
    SplitSampler.OptiSim: "OptiSim diverse clusters of fingerprints; each cluster stays in one set.",
    SplitSampler.TargetProperty: "Ordered by target value: extreme values go to the test "
    "set (deterministic).",
}


def sampler_widgets(problem_type: ProblemType, split_method: SplitMethod) -> None:
    """
    astartes sampler and, for fingerprint samplers, the sampling fingerprint.

    Only the samplers valid for ``problem_type`` and ``split_method`` are
    offered (``available_samplers``). The sampling fingerprint is used only to
    split the data; it does not change the representations.
    """
    options = available_samplers(problem_type, split_method)
    # A sampler chosen earlier may no longer be valid (e.g. after switching to
    # stratified); fall back to random instead of keeping an invalid choice.
    if st.session_state.get(GeneralConfigStateKeys.Sampler) not in options:
        st.session_state[GeneralConfigStateKeys.Sampler] = SplitSampler.Random
    sampler = st.selectbox(
        "Select the sampler",
        options=options,
        index=0,
        key=GeneralConfigStateKeys.Sampler,
        format_func=lambda s: s.value,
        help="astartes sampler that draws the training, validation and test sets "
        "(astartes default hyperparameters). "
        + " ".join(f"{s.value}: {_SAMPLER_HELP[s]}" for s in options),
    )
    st.caption(_SAMPLER_HELP[sampler])

    if sampler not in SAMPLERS_USING_FINGERPRINTS:
        return
    st.markdown(
        "**Sampling fingerprint** — each monomer's fingerprint is weighted by its ratio, as "
        "in the representations. Used only to split the data; it does not change the "
        "representations."
    )
    cols = st.columns(3)
    with cols[0]:
        fingerprint = st.selectbox(
            "Fingerprint",
            options=[
                MolecularDescriptor.Morgan,
                MolecularDescriptor.RDKitFP,
                MolecularDescriptor.PolyBERT,
            ],
            key=GeneralConfigStateKeys.SamplingFingerprint,
            format_func=lambda f: f.value,
        )
    if fingerprint == MolecularDescriptor.PolyBERT:
        st.caption(
            f"polyBERT embedding ({POLYBERT_MODEL}, downloaded on first use). It expects "
            "PSMILES; plain SMILES are embedded as given."
        )
        return
    with cols[1]:
        st.number_input(
            "Fingerprint size",
            min_value=1,
            value=MorganFingerprintConfig().fp_size,
            step=1,
            key=GeneralConfigStateKeys.SamplingFpSize,
        )
    with cols[2]:
        if fingerprint == MolecularDescriptor.Morgan:
            st.number_input(
                "Radius",
                min_value=0,
                value=MorganFingerprintConfig().radius,
                step=1,
                key=GeneralConfigStateKeys.SamplingFpRadius,
            )


def sampling_fingerprint_from_state() -> dict | None:
    """The ``sampling_fingerprint`` settings chosen in ``sampler_widgets`` (``None`` if unused)."""
    sampler = st.session_state.get(GeneralConfigStateKeys.Sampler, SplitSampler.Random)
    if sampler not in SAMPLERS_USING_FINGERPRINTS:
        return None
    fingerprint = st.session_state.get(
        GeneralConfigStateKeys.SamplingFingerprint, MolecularDescriptor.Morgan
    )
    if fingerprint == MolecularDescriptor.PolyBERT:
        return {"fingerprint": fingerprint}
    settings = {
        "fingerprint": fingerprint,
        "fp_size": st.session_state.get(
            GeneralConfigStateKeys.SamplingFpSize, RDKitFingerprintConfig().fp_size
        ),
    }
    if fingerprint == MolecularDescriptor.Morgan:
        settings["radius"] = st.session_state.get(
            GeneralConfigStateKeys.SamplingFpRadius, MorganFingerprintConfig().radius
        )
    return settings


def split_data_form(problem_type: ProblemType) -> bool:
    """
    Data splitting widgets.

    Returns
    -------
    bool
        Whether the chosen ratios leave data for training.
    """

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
                help="Proportion of the minority class after undersampling the majority "
                "class. The training and validation sets are each balanced after the "
                "split; the test set keeps the original class distribution (ACS Appl. "
                "Mater. Interfaces 2023, 15 (11), 14155–14163).",
            )

    else:
        st.selectbox(
            "Select a method to split the data",
            options=[SplitMethod.Random],
            index=0,
            disabled=True,
            key=GeneralConfigStateKeys.SplitMethod,
        )

    sampler_widgets(
        problem_type, st.session_state.get(GeneralConfigStateKeys.SplitMethod, SplitMethod.Random)
    )

    if split_type == SplitType.TrainValTest:
        n_repetitions = st.select_slider(
            "Select the number of bootstrap iterations",
            options=list(range(1, 11)),
            value=1,
            key=GeneralConfigStateKeys.BootstrapIterations,
        )
        sampler = st.session_state.get(GeneralConfigStateKeys.Sampler, SplitSampler.Random)
        if sampler in DETERMINISTIC_SAMPLERS and n_repetitions > 1:
            st.warning(deterministic_sampler_warning(sampler, n_repetitions))

    test_ratio = st.slider(
        "Select the test split ratio",
        min_value=0.01,
        max_value=0.9,
        value=0.2,
        key=GeneralConfigStateKeys.TestSize,
        help="Fraction of the full dataset held out for testing.",
    )

    val_ratio = st.slider(
        "Select the validation split ratio",
        min_value=0.01,
        max_value=0.9,
        value=0.2,
        key=GeneralConfigStateKeys.ValidationSize,
        help="Fraction of the full dataset used for validation (e.g. test 0.1 and "
        "validation 0.1 give an 80/10/10 split).",
    )

    train_ratio = 1.0 - test_ratio - val_ratio
    if train_ratio <= 0:
        st.error(
            f"Test ({test_ratio:.0%}) and validation ({val_ratio:.0%}) ratios add up to "
            f"{test_ratio + val_ratio:.0%}, leaving no data for training. Their sum must be "
            "below 100%."
        )
        return False

    st.caption(
        f"Split: {train_ratio:.0%} training / {val_ratio:.0%} validation / "
        f"{test_ratio:.0%} test of the full dataset."
    )
    return True
