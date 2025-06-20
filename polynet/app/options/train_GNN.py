from dataclasses import dataclass

from polynet.options.enums import ApplyWeightingToGraph


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
    ApplyMonomerWeighting: ApplyWeightingToGraph
