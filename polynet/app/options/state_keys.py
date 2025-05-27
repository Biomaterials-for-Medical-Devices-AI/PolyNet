from enum import StrEnum


class CreateExperimentStateKeys(StrEnum):
    """
    Enum for the keys used in the state of the CreateExperiment component.
    """

    ExperimentName = "ExperimentName"
    DatasetName = "DatasetName"
    SmilesCols = "SmilesCols"
    CanonicaliseSMILES = "CanonicaliseSMILES"
    IDCol = "IDCol"
    TargetVariableCol = "TargetVariableCol"
    ProblemType = "ProblemType"
    NumClasses = "NumClasses"
    TargetVariableName = "TargetVariableName"
    TargetVariableUnits = "TargetVariableUnits"
    EditPlot = "EditPlot"


class ViewExperimentKeys(StrEnum):
    ExperimentName = "ExperimentName"


class DescriptorCalculationStateKeys(StrEnum):
    """
    Enum for the keys used in the state of the DescriptorCalculation component.
    """

    MergeDescriptorsApproach = "MergeDescriptorsApproach"
    WeightingFactor = "WeightingFactor"
    DescriptorsDF = "DescriptorsDF"
    MergeDescriptors = "MergeDescriptors"
    GraphWeightingFactor = "GraphWeightingFactor"
    DescriptorsRDKit = "DescriptorsRDKit"
    SelectAllRDKitDescriptors = "SelectAllRDKitDescriptors"
    IndependentDFDescriptors = "IndependentDFDescriptors"
    IndependentRDKitDescriptors = "IndependentRDKitDescriptors"
    AtomProperties = "AtomProperties"
    BondProperties = "BondProperties"


class PlotOptionsStateKeys(StrEnum):
    """
    Enum for the keys used in the state of the PlotOptions component.
    """

    PlotAxisFontSize = "PlotAxisFontSize"
    PlotAxisTickSize = "PlotAxisTickSize"
    PlotColourScheme = "PlotColourScheme"
    DPI = "DPI"
    AngleRotateXaxisLabels = "AngleRotateXaxisLabels"
    AngleRotateYaxisLabels = "AngleRotateYaxisLabels"
    SavePlots = "SavePlots"
    PlotTitleFontSize = "PlotTitleFontSize"
    PlotFontFamily = "PlotFontFamily"
    Height = "Height"
    Width = "Width"


class TrainTMLStateKeys(StrEnum):
    TrainTML = "TrainTML"


class TrainGNNStateKeys(StrEnum):
    """
    Enum for the keys used in the state of the TrainGNN component.
    """

    TrainGNN = "TrainGNN"
    GNNConvolutionalLayers = "GNNConvolutionalLayers"
    GNNNumberOfLayers = "GNNNumberOfLayers"
    GNNEmbeddingDimension = "GNNEmbeddingDimension"
    GNNPoolingMethod = "GNNPoolingMethod"
    GNNReadoutLayers = "GNNReadoutLayers"
    GNNDropoutRate = "GNNDropoutRate"
    GNNLearningRate = "GNNLearningRate"
    GNNBatchSize = "GNNBatchSize"

    # Specific GNN Hyperparameters

    # GCN
    Improved = "Improved"


class GeneralConfigStateKeys(StrEnum):
    """
    Enum for the keys used in the state of the SplitData component.
    """

    SplitData = "SplitData"
    SplitMethod = "SplitMethod"
    TrainSize = "TrainSize"
    TestSize = "TestSize"
    ValidationSize = "ValidationSize"
    RandomSeed = "RandomSeed"

    Stratify = "Stratify"
    Shuffle = "Shuffle"
