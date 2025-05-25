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

    DescriptorsDF = "DescriptorsDF"
    DescriptorsRDKit = "DescriptorsRDKit"
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
