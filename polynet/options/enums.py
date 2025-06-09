from enum import StrEnum


class ProblemTypes(StrEnum):
    Classification = "classification"
    Regression = "regression"


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
