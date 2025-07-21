from dataclasses import dataclass


@dataclass
class TrainTMLOptions:
    """
    Options for training traditional machine learning models.
    """

    TransformFeatures: str = "NoTransformation"
    TrainLinearRegression: bool = True
    TrainLogisticRegression: bool = True
    TrainRandomForest: bool = True
    TrainSupportVectorMachine: bool = True
    TrainXGBoost: bool = True
