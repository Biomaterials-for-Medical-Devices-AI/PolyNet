import re

import pandas as pd
import streamlit as st

from polynet.utils.statistical_analysis import mcnemar_pvalue_matrix, regression_pvalue_matrix
from polynet.app.options.data import DataOptions
from polynet.app.options.general_experiment import GeneralConfigOptions
from polynet.app.options.state_keys import AnalyseResultsStateKeys, PlotCustomiserStateKeys
from polynet.app.options.train_GNN import TrainGNNOptions
from polynet.app.utils import (
    get_iterator_name,
    get_predicted_label_column_name,
    get_true_label_column_name,
)
from polynet.options.enums import DataSets, Plots, Results, ProblemTypes
from polynet.utils.plot_utils import plot_confusion_matrix, plot_parity, plot_pvalue_matrix


def compare_predictions_form(
    predictions_df: pd.DataFrame, target_variable_name: str, data_options: DataOptions
):

    st.write("### Model predictions pairwise comparison")

    label_col_name = get_true_label_column_name(target_variable_name=target_variable_name)
    true_vals = predictions_df[label_col_name].to_numpy()

    trained_models = [
        col.split(" ")[0] for col in predictions_df.columns if Results.Predicted in col
    ]
    compare_models = st.multiselect(
        "Select models to test statistically difference in predictions", options=trained_models
    )

    if len(compare_models) > 1:

        predicted_cols = [
            get_predicted_label_column_name(
                target_variable_name=target_variable_name, model_name=model
            )
            for model in compare_models
        ]
        predictions_array = predictions_df[predicted_cols].to_numpy().T

        if data_options.problem_type == ProblemTypes.Classification:
            p_matrix = mcnemar_pvalue_matrix(y_true=true_vals, predictions=predictions_array)
        elif data_options.problem_type == ProblemTypes.Regression:
            p_matrix = regression_pvalue_matrix(y_true=true_vals, predictions=predictions_array)

        plot = plot_pvalue_matrix(p_matrix=p_matrix, model_names=compare_models)

        return plot


def get_plot_customisation_form(
    plot_type: str,
    data: pd.DataFrame,
    data_options: DataOptions,
    color_by_opts=None,
    model_pred_col: str = None,
    model_true_cols: str = None,
):
    """
    Renders a dynamic Streamlit form for customizing a plot, based on plot_type.
    Returns a dictionary of kwargs for the plotting function.
    """
    config = {}

    with st.expander("Customise Plot", expanded=False):

        config["title"] = st.text_input(
            "Plot Title",
            value=f"{data_options.target_variable_name} {plot_type.title()} Plot",
            key=PlotCustomiserStateKeys.PlotTitle,
            help="Enter a title for the plot.",
        )

        config["title_fontsize"] = st.slider(
            "Title Font Size",
            min_value=8,
            max_value=30,
            value=16,
            step=1,
            key=PlotCustomiserStateKeys.PlotFontSize,
            help="Select the font size for the plot title.",
        )

        columns = st.columns(2)
        with columns[0]:
            config["x_label"] = st.text_input(
                "X-axis Label",
                value=model_true_cols,
                key=PlotCustomiserStateKeys.PlotXLabel,
                help="Enter a label for the X-axis.",
            )
            config["x_label_fontsize"] = st.slider(
                "X-axis Label Font Size",
                min_value=8,
                max_value=30,
                value=14,
                step=1,
                key=PlotCustomiserStateKeys.PlotXLabelFontSize,
                help="Select the font size for the X-axis label.",
            )
        with columns[1]:
            config["y_label"] = st.text_input(
                "Y-axis Label",
                value=model_pred_col,
                key=PlotCustomiserStateKeys.PlotYLabel,
                help="Enter a label for the Y-axis.",
            )
            config["y_label_fontsize"] = st.slider(
                "Y-axis Label Font Size",
                min_value=8,
                max_value=30,
                value=14,
                step=1,
                key=PlotCustomiserStateKeys.PlotYLabelFontSize,
                help="Select the font size for the Y-axis label.",
            )

        config["tick_size"] = st.select_slider(
            "Tick Size",
            options=list(range(8, 21)),
            value=12,
            key=PlotCustomiserStateKeys.TickSize,
            help="Select the font size for the tick labels.",
        )

        if plot_type == Plots.Parity.value:

            config["grid"] = st.checkbox(
                "Show Grid",
                value=True,
                key=PlotCustomiserStateKeys.PlotGrid,
                help="Check to display grid lines on the plot.",
            )

            config["point_size"] = st.slider(
                "Point Size",
                min_value=1,
                max_value=100,
                value=50,
                step=1,
                key=PlotCustomiserStateKeys.PlotPointSize,
                help="Select the size of the points in the plot.",
            )

            config["border_color"] = st.color_picker(
                "Point Border Color",
                value="#000000",
                key=PlotCustomiserStateKeys.PlotPointBorderColour,
                help="Select a color for the border of the plot.",
            )

            config["legend"] = st.checkbox(
                "Show Legend",
                value=True,
                key=PlotCustomiserStateKeys.ShowLegend,
                help="Check to display the legend on the plot.",
            )

            if config["legend"]:
                config["legend_fontsize"] = st.slider(
                    "Legend Font Size",
                    min_value=8,
                    max_value=20,
                    value=12,
                    step=1,
                    key=PlotCustomiserStateKeys.LegendFontSize,
                    help="Select the font size for the legend.",
                )
            else:
                config["legend_fontsize"] = None

            if len(color_by_opts) > 1:

                hue = st.selectbox(
                    "Color by",
                    options=color_by_opts,
                    key=PlotCustomiserStateKeys.ColourBy,
                    index=1,
                    help="Select a column to color the points in the plot.",
                )
                config["hue"] = data[hue] if hue else None

                style_by = st.selectbox(
                    "Style by",
                    options=color_by_opts,
                    key=PlotCustomiserStateKeys.StyleBy,
                    index=0,
                    help="Select a column to style the points in the plot.",
                )
                config["style_by"] = data[style_by] if style_by else None
            else:
                config["hue"] = None
                config["style_by"] = None

            if config["hue"] is not None:
                cmap = st.selectbox(
                    "Color Map",
                    options=["tab10", "pastel", "colorblind", "Spectral", "Custom"],
                    key=PlotCustomiserStateKeys.CMap,
                    help="Select a color map for the points in the plot.",
                )
                if cmap == "Custom":
                    cmap = []
                    for cat in config["hue"].unique():
                        color = st.color_picker(
                            f"Color for {cat}",
                            value="#1f77b4",
                            key=f"color_{cat}",
                            help=f"Select a color for the category '{cat}' in the plot.",
                        )
                        cmap.append(color)
                config["palette"] = cmap
                config["point_color"] = None
            else:
                config["palette"] = None
                config["point_color"] = st.color_picker(
                    "Point Fill Color",
                    value="#1f77b4",
                    key="point_color",
                    help="Select a color for the points in the plot.",
                )

        elif plot_type == Plots.ConfusionMatrix.value:
            config["cmap"] = st.selectbox(
                "Color Map",
                options=["Blues", "Greens", "Oranges"],
                key=PlotCustomiserStateKeys.CMap,
            )

            config["label_fontsize"] = st.slider(
                "Label Font Size",
                min_value=8,
                max_value=30,
                value=16,
                step=1,
                key=PlotCustomiserStateKeys.PlotLabelFontSize,
                help="Select the font size for the labels in the confusion matrix.",
            )

            labels = data_options.class_names.keys()
            config["display_labels"] = []
            for label in labels:
                name = st.text_input(
                    f"Name for label for {label}",
                    value=data_options.class_names[label],
                    key=f"{PlotCustomiserStateKeys.LabelNames}_{label}",
                    help=f"Enter the label for the class '{label}'.",
                )
                config["display_labels"].append(name)

        config["dpi"] = st.slider(
            "DPI",
            min_value=100,
            max_value=600,
            value=300,
            step=10,
            key="dpi",
            help="Select the DPI for the plot.",
        )

    return config


def confusion_matrix_plot_form(
    predictions_df: pd.DataFrame,
    general_experiment_options: GeneralConfigOptions,
    gnn_training_options: TrainGNNOptions,
    data_options: DataOptions,
):
    """
    Renders a Streamlit form for customizing and displaying a parity plot based on model predictions.
    Returns a matplotlib figure object.
    """

    iterator = get_iterator_name(general_experiment_options.split_type)

    iteration = st.multiselect(
        f"Select the {iterator} to display parity plot",
        options=predictions_df[iterator].unique(),
        key=AnalyseResultsStateKeys.PlotIterations,
        default=predictions_df[iterator].unique().tolist()[0],
        help="Select the iteration or fold for which you want to display the parity plot.",
    )

    trained_gnns = gnn_training_options.GNNConvolutionalLayers.keys()
    model_name = st.multiselect(
        "Select the model to display the confusion matrix",
        options=trained_gnns,
        default=list(trained_gnns)[0],
        key=AnalyseResultsStateKeys.PlotModels,
        help="Select the model for which you want to display the parity plot.",
    )

    model_pred_cols = [
        get_predicted_label_column_name(
            target_variable_name=data_options.target_variable_name, model_name=name
        )
        for name in model_name
    ]

    model_true_cols = get_true_label_column_name(
        target_variable_name=data_options.target_variable_name
    )

    set_name = st.multiselect(
        "Select the set to display parity plot",
        options=predictions_df[Results.Set.value].unique(),
        key=AnalyseResultsStateKeys.PlotSet,
        default=[DataSets.Test.value],
        help="Select the set for which you want to display the parity plot.",
    )

    plot_data = reshape_predictions(
        df=predictions_df,
        pred_cols=model_pred_cols,
        target_variable_name=data_options.target_variable_name,
    )

    model_pred_col = get_predicted_label_column_name(
        target_variable_name=data_options.target_variable_name, model_name=None
    )

    plot_data = plot_data[
        (plot_data[iterator].isin(iteration))
        & (plot_data[Results.Set.value].isin(set_name))
        & (plot_data[Results.Model.value].isin(model_name))
    ]

    with st.expander("Show Data", expanded=False):
        st.markdown(
            f"**Showing data for model: {model_name}, set: {set_name}, iterations: {iteration}**"
        )
        st.markdown(
            f"**True label column:** `{model_true_cols}` | **Predicted label column:** `{model_pred_col}`**"
        )
        st.dataframe(plot_data)

    plot_config = get_plot_customisation_form(
        plot_type=Plots.ConfusionMatrix.value,
        data=plot_data,
        data_options=data_options,
        color_by_opts=[None],
        model_pred_col=model_pred_col,
        model_true_cols=model_true_cols,
    )

    if not plot_data.empty:

        # invert preds and true to match x and y axes title
        fig = plot_confusion_matrix(
            y_true=plot_data[model_pred_col], y_pred=plot_data[model_true_cols], **plot_config
        )

        return fig


def parity_plot_form(
    predictions_df: pd.DataFrame,
    general_experiment_options: GeneralConfigOptions,
    gnn_training_options: TrainGNNOptions,
    data_options: DataOptions,
):

    color_by_opts = [None]

    iterator = get_iterator_name(general_experiment_options.split_type)

    iteration = st.multiselect(
        f"Select the {iterator} to display parity plot",
        options=predictions_df[iterator].unique(),
        key=AnalyseResultsStateKeys.PlotIterations,
        default=predictions_df[iterator].unique().tolist()[0],
        help="Select the iteration or fold for which you want to display the parity plot.",
    )

    if len(iteration) > 1:
        color_by_opts.append(iterator)

    trained_gnns = gnn_training_options.GNNConvolutionalLayers.keys()

    model_name = st.multiselect(
        "Select the model to display parity plot",
        options=trained_gnns,
        default=list(trained_gnns)[0],
        key=AnalyseResultsStateKeys.PlotModels,
        help="Select the model for which you want to display the parity plot.",
    )

    if len(model_name) > 1:
        color_by_opts.append(Results.Model.value)

    model_pred_cols = [
        get_predicted_label_column_name(
            target_variable_name=data_options.target_variable_name, model_name=name
        )
        for name in model_name
    ]

    model_true_cols = get_true_label_column_name(
        target_variable_name=data_options.target_variable_name
    )

    set_name = st.multiselect(
        "Select the set to display parity plot",
        options=predictions_df[Results.Set.value].unique(),
        key=AnalyseResultsStateKeys.PlotSet,
        default=[DataSets.Test.value],
        help="Select the set for which you want to display the parity plot.",
    )

    if len(set_name) > 1:
        color_by_opts.append(Results.Set.value)

    plot_data = reshape_predictions(
        df=predictions_df,
        pred_cols=model_pred_cols,
        target_variable_name=data_options.target_variable_name,
    )

    model_pred_col = get_predicted_label_column_name(
        target_variable_name=data_options.target_variable_name, model_name=None
    )

    plot_data = plot_data[
        (plot_data[iterator].isin(iteration))
        & (plot_data[Results.Set.value].isin(set_name))
        & (plot_data[Results.Model.value].isin(model_name))
    ]

    with st.expander("Show Data", expanded=False):
        st.markdown(
            f"**Showing data for model: {model_name}, set: {set_name}, iterations: {iteration}**"
        )
        st.markdown(
            f"**True label column:** `{model_true_cols}` | **Predicted label column:** `{model_pred_col}`**"
        )
        st.dataframe(plot_data)

    plot_config = get_plot_customisation_form(
        plot_type=Plots.Parity.value,
        data=plot_data,
        data_options=data_options,
        color_by_opts=color_by_opts,
        model_pred_col=model_pred_col,
        model_true_cols=model_true_cols,
    )

    if not plot_data.empty:

        fig = plot_parity(
            y_true=plot_data[model_true_cols], y_pred=plot_data[model_pred_col], **plot_config
        )

        return fig


def extract_model_name_and_target(col_name, target_variable_name):
    """Extract model name from column like 'Model Predicted target'."""
    pattern = rf"^(.*?)\s+Predicted\s+{re.escape(target_variable_name)}$"
    match = re.match(pattern, col_name)
    if match:
        return match.group(1)
    return None


def reshape_predictions(
    df: pd.DataFrame, pred_cols: list, target_variable_name: str
) -> pd.DataFrame:
    # Columns to keep (everything that’s not a prediction column)
    id_vars = [col for col in df.columns if col not in pred_cols]

    # Melt the DataFrame
    melted_df = df.melt(
        id_vars=id_vars, value_vars=pred_cols, var_name="Prediction_Column", value_name="Prediction"
    )

    # Extract model name from the column name
    melted_df[Results.Model.value] = melted_df["Prediction_Column"].apply(
        lambda col: extract_model_name_and_target(col, target_variable_name)
    )

    new_col_name = get_predicted_label_column_name(
        target_variable_name=target_variable_name, model_name=None
    )

    # Create new column name (without model name)
    melted_df[new_col_name] = melted_df["Prediction"]

    # Drop the old column
    melted_df = melted_df.drop(columns=["Prediction_Column", "Prediction"])

    return melted_df
