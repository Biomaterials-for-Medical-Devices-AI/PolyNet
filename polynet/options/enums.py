from enum import StrEnum


class ProblemTypes(StrEnum):
    Classification = "classification"
    Regression = "regression"


class Networks(StrEnum):
    GCN = "gcn"
    GAT = "gat"
    GraphSAGE = "graphsage"


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
