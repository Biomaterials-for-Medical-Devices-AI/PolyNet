from enum import StrEnum


class ProblemTypes(StrEnum):
    Classification = "classification"
    Regression = "regression"


class Networks(StrEnum):
    GCN = "gcn"
    GCNClassifier = "gcn_classifier"
    GCNRegressor = "gcn_regressor"
    TransformerGNN = "transformer_gnn"
    TransformerGNNClassifier = "transformer_gnn_classifier"
    TransformerGNNRegressor = "transformer_gnn_regressor"
    GAT = "gat"
    GraphSAGE = "graphsage"


class NetworkParams(StrEnum):
    # GCN
    Improved = "Improved"
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
    GlobalMaxPool = "gmp"
    GlobalAddPool = "gsp"
    GlobalMeanPool = "gmeanp"
    GlobalMeanMaxPool = "gmeanmaxp"


class Split_types(StrEnum):
    TrainValTest = "train_val_test"
    TrainTest = "train_test"
    CrossValidation = "cross_validation"
    NestedCrossValidation = "nested_cross_validation"
    LeaveOneOut = "leave_one_out"


class SplitMethods(StrEnum):
    Random = "random"
    Stratified = "stratified"


class DataSets(StrEnum):
    Training = "Train"
    Validation = "Validation"
    Test = "Test"


class Results(StrEnum):
    Label = "True"
    Predicted = "Predicted"


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
