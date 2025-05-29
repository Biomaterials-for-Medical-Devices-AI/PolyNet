from enum import StrEnum


class ProblemTypes(StrEnum):
    Classification = "classification"
    Regression = "regression"


class Networks(StrEnum):
    GCN = "GCN"
    GCNClassifier = "GCN Classifier"
    GCNRegressor = "GCN Regressor"
    TransformerGNN = "TransformerConvGNN"
    TransformerGNNClassifier = "transformer_gnn_classifier"
    TransformerGNNRegressor = "transformer_gnn_regressor"
    GAT = "gat"
    GraphSAGE = "graphsage"


class NetworkParams(StrEnum):
    # GCN
    Improved = "improved"
    # TransformerGNN
    NumHeads = "NumHeads"


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


class Split_types(StrEnum):
    TrainValTest = "train_val_test"
    TrainTest = "train_test"
    CrossValidation = "cross_validation"
    NestedCrossValidation = "nested_cross_validation"
    LeaveOneOut = "leave_one_out"


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
