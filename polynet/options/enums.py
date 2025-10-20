from enum import StrEnum


class ProblemTypes(StrEnum):
    Classification = "classification"
    Regression = "regression"


class StringRepresentation(StrEnum):
    Smiles = "SMILES"
    PSmiles = "Psmiles"


class Networks(StrEnum):
    GCN = "GCN"
    TransformerGNN = "TransformerConvGNN"
    GAT = "GAT"
    GraphSAGE = "GraphSAGE"
    MPNN = "MPNN"
    CGGNN = "CGGNN"


class NetworkParams(StrEnum):
    # GCN
    Improved = "improved"
    # TransformerGNN
    NumHeads = "NumHeads"
    # GAT

    # GraphSAGE
    Bias = "bias"
    # MPNN

    # shared params
    Pooling = "pooling"
    NumConvolutions = "n_convolutions"
    EmbeddingDim = "embedding_dim"
    ReadoutLayers = "readout_layers"
    Dropout = "dropout"
    ApplyWeightingGraph = "apply_weighting_to_graph"


class ExplainAlgorithms(StrEnum):
    GNNExplainer = "GNNExplainer"
    IntegratedGradients = "IntegratedGradients"
    Saliency = "Saliency"
    InputXGradients = "InputXGradients"
    Deconvolution = "Deconvolution"
    ShapleyValueSampling = "ShapleyValueSampling"
    GuidedBackprop = "GuidedBackprop"


class Optimizers(StrEnum):
    Adam = "adam"
    SGD = "sgd"
    RMSprop = "rmsprop"
    Adadelta = "adadelta"
    Adagrad = "adagrad"


class Schedulers(StrEnum):
    StepLR = "steplr"
    MultiStepLR = "multisteplr"
    ExponentialLR = "exponentiallr"
    ReduceLROnPlateau = "reducelronplateau"


class ApplyWeightingToGraph(StrEnum):
    BeforeMPP = "Before MPP"
    BeforePooling = "Before Pooling"
    NoWeighting = "No Weighting"


class Pooling(StrEnum):
    GlobalMaxPool = "GlobalMaxPool"
    GlobalAddPool = "GlobalAddPool"
    GlobalMeanPool = "GlobalMeanPool"
    GlobalMeanMaxPool = "GlobalMeanMaxPool"


class SplitTypes(StrEnum):
    TrainValTest = "train_val_test"
    TrainTest = "train_test"
    CrossValidation = "cross_validation"
    NestedCrossValidation = "nested_cross_validation"
    LeaveOneOut = "leave_one_out"


class IteratorTypes(StrEnum):
    BootstrapIteration = "Bootstrap Iteration"
    CrossValidation = "Cross Validation"
    Iteration = "Iteration"
    Fold = "Fold"
    LeaveOneOut = "Leave One Out"


class SplitMethods(StrEnum):
    Random = "Random"
    Stratified = "Stratified"


class DataSets(StrEnum):
    Training = "Train"
    Validation = "Validation"
    Test = "Test"


class Results(StrEnum):
    Label = "True"
    Predicted = "Predicted"
    Index = "Index"
    Set = "Set"
    Score = "Score"
    Model = "Model"
    Loaders = "Loaders"


class EvaluationMetrics(StrEnum):
    Accuracy = "Accuracy"
    Precision = "Precision"
    Recall = "Recall"
    F1Score = "F1 Score"
    AUROC = "AUROC"
    MCC = "MCC"
    Specificity = "Specificity"
    GScore = "G-Score"
    RMSE = "RMSE"
    MAE = "MAE"
    R2 = "R2"


class Plots(StrEnum):
    Parity = "Parity"
    ConfusionMatrix = "Confusion Matrix"
    ROC = "ROC Curve"
    PrecisionRecall = "Precision-Recall Curve"
    Loss = "Loss"
    TrainingHistory = "Training History"
    FeatureImportance = "Feature Importance"
    GraphVisualization = "Graph Visualization"
    MatrixPlot = "matrix_plot"
    MetricsBoxPlot = "metrics_box_plot"


class AtomBondDescriptorDictKeys(StrEnum):
    AllowableVals = "allowable_vals"
    Wildcard = "wildcard"
    Options = "options"
    Default = "default"
    Description = "description"


class AtomFeatures(StrEnum):
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


class BondFeatures(StrEnum):
    GetBondTypeAsDouble = "GetBondTypeAsDouble"
    GetIsAromatic = "GetIsAromatic"
    GetIsConjugated = "GetIsConjugated"
    GetStereo = "GetStereo"
    IsInRing = "IsInRing"


class DescriptorMergingMethods(StrEnum):
    Average = "Average"
    WeightedAverage = "Weighted Average"
    Concatenate = "Concatenate"
    NoMerging = "No Merging"


class ImportanceNormalisationMethods(StrEnum):
    Local = "Local"
    Global = "Global"
    NoNormalisation = "No Normalisation"


class DimensionalityReduction(StrEnum):
    tSNE = "t-SNE"
    Perplexity = "perplexity"
    PCA = "PCA"


class TradtionalMLModels(StrEnum):
    LinearRegression = "Linear Regression"
    LogisticRegression = "Logistic Regression"
    RandomForest = "Random Forest"
    XGBoost = "XGBoost"
    SupportVectorMachine = "Support Vector Machine"
    KNeighborsClassifier = "K-Neighbors Classifier"
    DecisionTreeClassifier = "Decision Tree Classifier"


class TransformDescriptors(StrEnum):
    NoTransformation = "No Transformation"
    StandardScaler = "Standard Scaler"
    MinMaxScaler = "Min-Max Scaler"
    RobustScaler = "Robust Scaler"
    PowerTransformer = "Power Transformer"
    QuantileTransformer = "Quantile Transformer"
    Normalizer = "Normalizer"


class StatisticalTests(StrEnum):
    McNemar = "mcnemar"
    Wilcoxon = "Wilcoxon"
    TTest = "t-test"
