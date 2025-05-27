from dataclasses import dataclass


@dataclass
class TrainGNNOptions:

    GNNConvolutionalLayers: dict[str, dict]
    GNNNumberOfLayers: int
    GNNEmbeddingDimension: int
    GNNPoolingMethod: str
    GNNReadoutLayers: int
    GNNDropoutRate: float
    GNNLearningRate: float
    GNNBatchSize: int
