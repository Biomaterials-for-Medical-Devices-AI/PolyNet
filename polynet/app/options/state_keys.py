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
    ClassNames = "ClassNames"
    TargetVariableName = "TargetVariableName"
    TargetVariableUnits = "TargetVariableUnits"
    EditPlot = "EditPlot"
    StringRepresentation = "StringRepresentation"


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
    polyBERTfp = "polyBERTfp"
    polyBERTindependent = "polyBERTindependent"
    SelectAllRDKitDescriptors = "SelectAllRDKitDescriptors"
    IndependentDFDescriptors = "IndependentDFDescriptors"
    IndependentRDKitDescriptors = "IndependentRDKitDescriptors"
    AtomProperties = "AtomProperties"
    BondProperties = "BondProperties"
    ShowFrequency = "ShowFrequency"


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
    PerformHyperparameterTuning = "PerformHyperparameterTuning"

    TrainLinearRegression = "LinearRegression"
    TrainLogisticRegression = "LogisticRegression"
    LinearRegressionFitIntercept = "LinearRegressionFitIntercept"

    LogisticRegressionPenalty = "LogissticRegressionPenalty"
    LogisticRegressionC = "LogisticRegressionC"
    LogisticRegressionSolver = "LogisticRegressionSolver"

    TrainRandomForest = "RandomForest"
    RFNumberEstimators = "RFNumberEstimators"
    RFMinSamplesSplit = "RFMinSamplesSplit"
    RFMinSamplesLeaf = "RFMinSamplesLeaf"
    RFMaxDepth = "RFMaxDepth"

    TrainXGBoost = "XGBoost"
    XGBNumberEstimators = "XGBNumberEstimators"
    XGBLearningRate = "XGBLearningRate"
    XGBSubsampleSize = "XGBSubsampleSize"
    XGBMaxDepth = "XGBMaxDepth"

    TrainSupportVectorMachine = "SupportVectorMachine"
    SVMKernel = "SVMKernel"
    SVMDegree = "SVMDegree"
    SVMC = "SVMC"

    TrasformFeatures = "TrasformFeatures"


class TrainGNNStateKeys(StrEnum):
    """
    Enum for the keys used in the state of the TrainGNN component.
    """

    TrainGNN = "TrainGNN"
    HypTunning = "HypTunning"
    SharedGNNParams = "SharedGNNParams"
    GNNConvolutionalLayers = "GNNConvolutionalLayers"
    GNNNumberOfLayers = "GNNNumberOfLayers"
    GNNEmbeddingDimension = "GNNEmbeddingDimension"
    GNNPoolingMethod = "GNNPoolingMethod"
    GNNReadoutLayers = "GNNReadoutLayers"
    GNNDropoutRate = "GNNDropoutRate"
    GNNLearningRate = "GNNLearningRate"
    GNNBatchSize = "GNNBatchSize"
    GNNMonomerWeighting = "GNNMonomerWeighting"
    AsymmetricLoss = "AsymmetricLoss"
    ImbalanceStrength = "ImbalanceStrength"

    # Specific GNN Hyperparameters

    # GCN
    Improved = "Improved"

    # TransformerGNN
    NumHeads = "NumHeads"

    # Gat
    NHeads = "NHeads"

    # GraphSAGE
    Bias = "Bias"


class GeneralConfigStateKeys(StrEnum):
    """
    Enum for the keys used in the state of the SplitData component.
    """

    SplitType = "SplitType"
    BalanceClasses = "BalanceClasses"
    DesiredProportion = "DesiredProportion"
    SplitMethod = "SplitMethod"
    TrainSize = "TrainSize"
    TestSize = "TestSize"
    ValidationSize = "ValidationSize"
    RandomSeed = "RandomSeed"
    BootstrapIterations = "BootstrapIterations"

    Stratify = "Stratify"
    Shuffle = "Shuffle"


class ExplainModelStateKeys(StrEnum):
    """
    Enum for the keys used in the state of the ExplainModel component.
    """

    ExplainModels = "ExplainModels"
    ExplainModel = "ExplainModel"
    ExplainSet = "ExplainSet"
    ExplainIDSelector = "ExplainIDSelector"
    ExplainManuallySelector = "ExplainManuallySelector"
    PlotIDSelector = "PlotIDSelector"
    SetMolName = "SetMolName"
    ExplainAlgorithm = "ExplainAlgorithm"
    ExplainNodeFeats = "ExplainNodeFeats"
    NegColorPlots = "NegColorPlots"
    PosColorPlots = "PosColorPlots"
    CutoffSelector = "CutoffSelector"
    NormalisationMethodSelector = "NormalisationMethodSelector"


class PredictPageStateKeys(StrEnum):
    PredictData = "PredictData"
    CompareTarget = "CompareTarget"
    SelectAllModels = "SelectAllModels"
    SelectModel = "SelectModel"


class ProjectionPlotStateKeys(StrEnum):
    CreateProjectionPlot = "CreateProjectionPlot"
    ModelForProjections = "ModelForProjections"
    DimensionReductionMethod = "DimensionReductionMethod"
    tSNEPerplexity = "tSNEPerplexity"
    ColourProjectionBy = "ColourProjectionBy"
    ProjectionColourMap = "ProjectionColourMap"
    ProjectionDescriptorSelector = "ProjectionDescriptorSelector"
    PlotProjection = "PlotProjection"
    PlotProjectionSet = "PlotProjectionSet"
    PlotProjectionMols = "PlotProjectionMols"
    ProjectionData = "ProjectionData"
    ProjectionManualSelection = "ProjectionManualSelection"


class AnalyseResultsStateKeys(StrEnum):
    """
    Enum for the keys used in the state of the AnalyseResults component.
    """

    PlotIterations = "PlotIterations"
    PlotModels = "PlotModels"
    PlotSet = "PlotSet"


class PlotCustomiserStateKeys(StrEnum):
    """
    Enum for the keys used in the state of the PlotCustomiser component.
    """

    PlotTitle = "PlotTitle"
    PlotFontSize = "PlotFontSize"
    PlotGrid = "PlotGrid"
    ShowLegend = "ShowLegend"
    LegendFontSize = "LegendFontSize"
    PlotXLabel = "PlotXLabel"
    PlotXLabelFontSize = "PlotXLabelFontSize"
    PlotYLabel = "PlotYLabel"
    PlotYLabelFontSize = "PlotYLabelFontSize"
    TickSize = "TickSize"

    PlotPointSize = "PlotPointSize"
    PlotPointBorderColour = "PlotPointBorderColour"

    ColourBy = "ColourBy"
    StyleBy = "StyleBy"

    DPI = "DPI"

    CMap = "CMap"
    PlotLabelFontSize = "PlotLabelFontSize"
    LabelNames = "LabelNames"
