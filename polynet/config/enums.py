"""
polynet.config.enums
====================
Single source of truth for all enumerations used across the polynet package.

These enums are shared between:
- The Streamlit app (app/)
- The YAML-based script entry point (config/from_yaml.py)
- The core pipeline (models/, training/, inference/, etc.)

All enums use ``StrEnum`` so that values serialise naturally to/from strings,
making YAML configs and UI labels human-readable without extra conversion steps.
"""

from enum import StrEnum

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class DatasetName(StrEnum):
    """Benchmarking datasets"""

    CuratedTg = "curated_tg"


class ProblemType(StrEnum):
    """Supervised learning task type."""

    Classification = "classification"
    Regression = "regression"


class StringRepresentation(StrEnum):
    """Polymer string representation format."""

    SMILES = "smiles"
    PSMILES = "psmiles"


class SplitType(StrEnum):
    """Top-level data splitting strategy."""

    TrainValTest = "train_val_test"
    TrainTest = "train_test"
    CrossValidation = "cross_validation"
    NestedCrossValidation = "nested_cross_validation"
    LeaveOneOut = "leave_one_out"


class SplitMethod(StrEnum):
    """How samples are assigned to splits."""

    Random = "random"
    Stratified = "stratified"


class IteratorType(StrEnum):
    """Labels for iterative training loops."""

    BootstrapIteration = "bootstrap_iteration"
    CrossValidation = "cross_validation"
    Iteration = "iteration"
    Fold = "fold"
    LeaveOneOut = "leave_one_out"


# ---------------------------------------------------------------------------
# Molecular representation
# ---------------------------------------------------------------------------


class MolecularDescriptor(StrEnum):
    """Molecular descriptor / feature set."""

    RDKit = "rdkit"
    PolyBERT = "polybert"
    DataFrame = "dataframe"
    RDKit_DataFrame = "rdkit_dataframe"
    PolyMetriX = "polymetrix"
    Morgan = "morgan"
    RDKitFP = "rdkitfp"


class PMXAggMethod(StrEnum):
    Sum = "sum"
    Mean = "mean"
    Max = "max"
    Min = "min"


class PMXChemFeature(StrEnum):
    """
    Chemical descriptor-based featurizers operating on molecular structures.

    The enum values follow the PolyNet configuration convention:
    lowercase with underscores.
    """

    NumHBondDonors = "num_hbond_donors"
    NumHBondAcceptors = "num_hbond_acceptors"
    NumRotatableBonds = "num_rotatable_bonds"
    NumRings = "num_rings"
    NumNonAromaticRings = "num_non_aromatic_rings"
    NumAromaticRings = "num_aromatic_rings"
    NumAtoms = "num_atoms"
    TopologicalSurfaceArea = "topological_surface_area"
    FractionBicyclicRings = "fraction_bicyclic_rings"
    NumAliphaticHeterocycles = "num_aliphatic_heterocycles"
    SlogPVSA1 = "slogpvsa1"
    BalabanJIndex = "balaban_j_index"
    MolecularWeight = "molecular_weight"
    Sp3CarbonCountFeaturizer = "sp3_carbon_count"
    Sp2CarbonCountFeaturizer = "sp2_carbon_count"
    MaxEStateIndex = "max_estate_index"
    SmrVSA5 = "smr_vsa5"
    FpDensityMorgan1 = "fp_density_morgan1"
    HalogenCounts = "halogen_counts"
    BondCounts = "bond_counts"
    BridgingRingsCount = "bridging_rings_count"
    MaxRingSize = "max_ring_size"
    HeteroatomCount = "heteroatom_count"
    HeteroatomDensity = "heteroatom_density"


class PMXTopoFeature(StrEnum):
    """
    Topology-aware featurizers for polymer representation.
    """

    NumSideChainFeaturizer = "num_sidechains"
    NumBackBoneFeaturizer = "num_backbone"
    SideChainLengthFeaturizer = "sidechain_length"


class AtomFeature(StrEnum):
    """RDKit atom-level features available for graph construction."""

    GetAtomicNum = "GetAtomicNum"
    GetTotalDegree = "GetTotalDegree"
    GetFormalCharge = "GetFormalCharge"
    GetTotalNumHs = "GetTotalNumHs"
    GetHybridization = "GetHybridization"
    GetChiralTag = "GetChiralTag"
    GetIsAromatic = "GetIsAromatic"
    GetMass = "GetMass"
    GetImplicitValence = "GetImplicitValence"
    IsInRing = "IsInRing"


class BondFeature(StrEnum):
    """RDKit bond-level features available for graph construction."""

    GetBondTypeAsDouble = "GetBondTypeAsDouble"
    GetIsAromatic = "GetIsAromatic"
    GetIsConjugated = "GetIsConjugated"
    GetStereo = "GetStereo"
    IsInRing = "IsInRing"


class DescriptorMergingMethod(StrEnum):
    """How per-monomer descriptors are merged into a single polymer representation."""

    Average = "average"
    WeightedAverage = "weighted_average"
    Concatenate = "concatenate"
    NoMerging = "no_merging"


class TransformDescriptor(StrEnum):
    """Feature scaling / transformation applied before traditional ML training."""

    NoTransformation = "no_transformation"
    StandardScaler = "standard_scaler"
    MinMaxScaler = "min_max_scaler"
    RobustScaler = "robust_scaler"
    PowerTransformer = "power_transformer"
    QuantileTransformer = "quantile_transformer"
    Normalizer = "normalizer"


class FeatureSelection(StrEnum):
    NoSelection = "no_selection"
    Variance = "variance"
    Correlation = "correlation"
    LASSO = "lasso"


class FragmentationMethod(StrEnum):
    """Molecular fragmentation strategies for graph-based explainability."""

    MurckoScaffold = "murcko_scaffold"
    BRICS = "brics"
    FunctionalGroups = "functional_groups"
    Recap = "recap"


# ---------------------------------------------------------------------------
# Graph neural network architecture
# ---------------------------------------------------------------------------


class Network(StrEnum):
    """Supported GNN architectures."""

    GCN = "GCN"
    TransformerGNN = "TransformerConvGNN"
    GAT = "GAT"
    GraphSAGE = "GraphSAGE"
    MPNN = "MPNN"
    CGGNN = "CGGNN"


class ArchitectureParam(StrEnum):
    """GNN architecture hyperparameter keys (structure-related)."""

    # GCN
    Improved = "improved"
    # TransformerGNN
    NumHeads = "num_heads"
    # GraphSAGE
    Bias = "bias"
    # Shared
    NumConvolutions = "n_convolutions"
    EmbeddingDim = "embedding_dim"
    ReadoutLayers = "readout_layers"
    Dropout = "dropout"
    PoolingMethod = "pooling"
    ApplyWeightingGraph = "apply_weighting_to_graph"


class TrainingParam(StrEnum):
    """GNN training hyperparameter keys (optimisation-related)."""

    LearningRate = "learning_rate"
    BatchSize = "batch_size"
    AsymmetricLossStrength = "asymmetric_loss_strength"
    Seed = "seed"


class Pooling(StrEnum):
    """Graph-level pooling strategies."""

    GlobalMaxPool = "global_max_pool"
    GlobalAddPool = "global_add_pool"
    GlobalMeanPool = "global_mean_pool"
    GlobalMeanMaxPool = "global_mean_max_pool"


class ApplyWeightingToGraph(StrEnum):
    """Where in the GNN forward pass to apply monomer weighting."""

    BeforeMPP = "before_mpp"
    BeforePooling = "before_pooling"
    NoWeighting = "no_weighting"


# ---------------------------------------------------------------------------
# Optimisation and scheduling
# ---------------------------------------------------------------------------


class Optimizer(StrEnum):
    """Gradient descent optimizers."""

    Adam = "adam"
    SGD = "sgd"
    RMSprop = "rmsprop"
    Adadelta = "adadelta"
    Adagrad = "adagrad"


class Scheduler(StrEnum):
    """Learning rate schedulers."""

    StepLR = "step_lr"
    MultiStepLR = "multi_step_lr"
    ExponentialLR = "exponential_lr"
    ReduceLROnPlateau = "reduce_lr_on_plateau"


# ---------------------------------------------------------------------------
# Traditional ML
# ---------------------------------------------------------------------------


class TraditionalMLModel(StrEnum):
    """Supported traditional machine learning models."""

    LinearRegression = "linear_regression"
    LogisticRegression = "logistic_regression"
    RandomForest = "random_forest"
    XGBoost = "xgboost"
    SupportVectorMachine = "support_vector_machine"
    KNeighborsClassifier = "k_neighbors_classifier"
    DecisionTreeClassifier = "decision_tree_classifier"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class EvaluationMetric(StrEnum):
    """Model evaluation metrics."""

    # Classification
    Accuracy = "accuracy"
    Precision = "precision"
    Recall = "recall"
    F1Score = "f1_score"
    AUROC = "auroc"
    MCC = "mcc"
    Specificity = "specificity"
    GScore = "g_score"
    # Regression
    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------


class ExplainAlgorithm(StrEnum):
    """Supported attribution / explainability algorithms."""

    GNNExplainer = "gnn_explainer"
    IntegratedGradients = "integrated_gradients"
    Saliency = "saliency"
    InputXGradients = "input_x_gradients"
    Deconvolution = "deconvolution"
    ShapleyValueSampling = "shapley_value_sampling"
    GuidedBackprop = "guided_backprop"


class ImportanceNormalisationMethod(StrEnum):
    """How node/edge importance scores are normalised for visualisation."""

    Local = "local"
    Global = "global"
    NoNormalisation = "no_normalisation"


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


class Plot(StrEnum):
    """Available plot types."""

    Parity = "parity"
    ConfusionMatrix = "confusion_matrix"
    ROC = "roc_curve"
    PrecisionRecall = "precision_recall_curve"
    Loss = "loss"
    TrainingHistory = "training_history"
    FeatureImportance = "feature_importance"
    GraphVisualization = "graph_visualization"
    MatrixPlot = "matrix_plot"
    MetricsBoxPlot = "metrics_box_plot"


class DimensionalityReduction(StrEnum):
    """Dimensionality reduction methods for embedding visualisation."""

    tSNE = "t_sne"
    PCA = "pca"


# ---------------------------------------------------------------------------
# Internal / featurizer
# ---------------------------------------------------------------------------


class AtomBondDescriptorDictKey(StrEnum):
    """Keys used in atom/bond feature descriptor dictionaries."""

    AllowableVals = "allowable_vals"
    Wildcard = "wildcard"
    Options = "options"
    Default = "default"
    Description = "description"
