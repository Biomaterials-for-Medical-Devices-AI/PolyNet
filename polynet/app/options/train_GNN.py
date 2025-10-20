from dataclasses import dataclass


@dataclass
class TrainGNNOptions:

    GNNConvolutionalLayers: dict[str, dict]
    HyperparameterOptimisation: bool = False
    ShareGNNParameters: bool = True
    TrainGNNModel: bool = True
