from dataclasses import dataclass


@dataclass
class TrainTMLOptions:
    """
    Options for training traditional machine learning models.
    """

    TrainTMLModels: bool = False
    TransformFeatures: str = "NoTransformation"
    HyperparameterOptimization: bool = False
    TMLModelsParams: dict[str, dict] = None
    TrainLinearRegression: bool = True
    TrainLogisticRegression: bool = True
    TrainRandomForest: bool = True
    TrainSupportVectorMachine: bool = True
    TrainXGBoost: bool = True
