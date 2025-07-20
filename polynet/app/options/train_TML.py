from dataclasses import dataclass


@dataclass
class TrainTMLOptions:
    """
    Options for training traditional machine learning models.
    """

    TrainLinearRegression: bool = True
    TrainLogisticRegression: bool = True
    TrainRandomForest: bool = True
    TrainXGBoost: bool = True
    TrainSupportVectorMachine: bool = True
