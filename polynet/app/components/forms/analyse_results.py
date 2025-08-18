import re

import pandas as pd
import streamlit as st

from polynet.utils.statistical_analysis import (
    mcnemar_pvalue_matrix,
    regression_pvalue_matrix,
    metrics_pvalue_matrix,
)
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
from polynet.utils.plot_utils import (
    plot_confusion_matrix,
    plot_parity,
    plot_pvalue_matrix,
    plot_bootstrap_boxplots,
)


def compare_predictions_form(
    predictions_df: pd.DataFrame, target_variable_name: str, data_options: DataOptions
):

    st.write("### Model predictions pairwise comparison")

    analyse_set = st.selectbox(
        "Select a set to analyse",
        options=[DataSets.Training, DataSets.Validation, DataSets.Test, "All sets"],
        index=2,
        key="analyse_set",
    )

    if analyse_set != "All sets":
        predictions_df = predictions_df.loc[predictions_df[Results.Set] == analyse_set]

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

        config = get_plot_customisation_form(
            plot_type=Plots.MatrixPlot,
            data=None,
            data_options=data_options,
            color_by_opts=None,
            model_pred_col=None,
            model_true_cols=None,
        )
        plot = plot_pvalue_matrix(p_matrix=p_matrix, model_names=compare_models, **config)

        return plot


def compare_metrics_form(metrics: dict, data_options: DataOptions = None):

    st.write("### Model metrics comparison")

    analyse_set = st.selectbox(
        "Select a set to analyse",
        options=[DataSets.Training, DataSets.Validation, DataSets.Test],
        index=2,
        key="analyse_set_metrics",
    )

    # Build: data_dict[model][metric_name] -> list of bootstrap values
    data_dict = {}
    metrics_name = set()
    for _, dictio in metrics.items():
        for model, per_model_dict in dictio.items():
            data_dict.setdefault(model, {})
            for metric_name, val in per_model_dict[analyse_set].items():
                data_dict[model].setdefault(metric_name, [])
                data_dict[model][metric_name].append(val)
                metrics_name.add(metric_name)

    models = st.multiselect(
        "Select models to compare", options=list(data_dict.keys()), key="comparemodelmetrics"
    )

    metric_choice = st.selectbox(
        "Select a metric to compare", options=sorted(metrics_name), key="metriccomparemetric"
    )

    selected = {m: data_dict[m][metric_choice] for m in models if m in data_dict}

    plot_type = st.selectbox("Select a type of plot", options=["P-value Matrix", "Box Plot"])

    if len(selected) > 1:

        if plot_type == "P-value Matrix":
            config = get_plot_customisation_form(
                plot_type=Plots.MatrixPlot,
                data=None,
                data_options=data_options,
                color_by_opts=None,
                model_pred_col=None,
                model_true_cols=None,
            )
            p_matrix, order = metrics_pvalue_matrix(selected, test="wilcoxon")
            plot = plot_pvalue_matrix(p_matrix=p_matrix, model_names=order, **config)
        elif plot_type == "Box Plot":
            config = get_plot_customisation_form(
                plot_type=Plots.MetricsBoxPlot,
                data=None,
                data_options=data_options,
                color_by_opts=None,
                model_pred_col=None,
                model_true_cols=None,
            )
            plot = plot_bootstrap_boxplots(
                metrics_dict=selected, metric_name=metric_choice, **config
            )
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
            key=PlotCustomiserStateKeys.PlotTitle + plot_type,
            help="Enter a title for the plot.",
        )

        config["title_fontsize"] = st.slider(
            "Title Font Size",
            min_value=8,
            max_value=30,
            value=16,
            step=1,
            key=PlotCustomiserStateKeys.PlotFontSize + plot_type,
            help="Select the font size for the plot title.",
        )

        columns = st.columns(2)
        with columns[0]:
            config["height"] = st.slider(
                "Plot height",
                min_value=4,
                max_value=14,
                value=7,
                step=1,
                key="plot_height" + plot_type,
            )
            if plot_type not in [Plots.MatrixPlot, Plots.MetricsBoxPlot]:
                config["x_label"] = st.text_input(
                    "X-axis Label",
                    value=model_true_cols,
                    key=PlotCustomiserStateKeys.PlotXLabel + plot_type,
                    help="Enter a label for the X-axis.",
                )
            config["x_label_fontsize"] = st.slider(
                "X-axis Label Font Size",
                min_value=8,
                max_value=30,
                value=14,
                step=1,
                key=PlotCustomiserStateKeys.PlotXLabelFontSize + plot_type,
                help="Select the font size for the X-axis label.",
            )
        with columns[1]:

            config["width"] = st.slider(
                "Plot width",
                min_value=4,
                max_value=14,
                value=7,
                step=1,
                key="plot_width" + plot_type,
            )

            if plot_type not in [Plots.MatrixPlot, Plots.MetricsBoxPlot]:
                config["y_label"] = st.text_input(
                    "Y-axis Label",
                    value=model_pred_col,
                    key=PlotCustomiserStateKeys.PlotYLabel + plot_type,
                    help="Enter a label for the Y-axis.",
                )
            config["y_label_fontsize"] = st.slider(
                "Y-axis Label Font Size",
                min_value=8,
                max_value=30,
                value=14,
                step=1,
                key=PlotCustomiserStateKeys.PlotYLabelFontSize + plot_type,
                help="Select the font size for the Y-axis label.",
            )

        config["tick_size"] = st.select_slider(
            "Tick Size",
            options=list(range(8, 21)),
            value=12,
            key=PlotCustomiserStateKeys.TickSize + plot_type,
            help="Select the font size for the tick labels.",
        )

        if plot_type == Plots.Parity.value:

            config["grid"] = st.checkbox(
                "Show Grid",
                value=True,
                key=PlotCustomiserStateKeys.PlotGrid + plot_type,
                help="Check to display grid lines on the plot.",
            )

            config["point_size"] = st.slider(
                "Point Size",
                min_value=1,
                max_value=100,
                value=50,
                step=1,
                key=PlotCustomiserStateKeys.PlotPointSize + plot_type,
                help="Select the size of the points in the plot.",
            )

            config["border_color"] = st.color_picker(
                "Point Border Color",
                value="#000000",
                key=PlotCustomiserStateKeys.PlotPointBorderColour + plot_type,
                help="Select a color for the border of the plot.",
            )

            config["legend"] = st.checkbox(
                "Show Legend",
                value=True,
                key=PlotCustomiserStateKeys.ShowLegend + plot_type,
                help="Check to display the legend on the plot.",
            )

            if config["legend"]:
                config["legend_fontsize"] = st.slider(
                    "Legend Font Size",
                    min_value=8,
                    max_value=20,
                    value=12,
                    step=1,
                    key=PlotCustomiserStateKeys.LegendFontSize + plot_type,
                    help="Select the font size for the legend.",
                )
            else:
                config["legend_fontsize"] = None

            if len(color_by_opts) > 1:

                hue = st.selectbox(
                    "Color by",
                    options=color_by_opts,
                    key=PlotCustomiserStateKeys.ColourBy + plot_type,
                    index=1,
                    help="Select a column to color the points in the plot.",
                )
                config["hue"] = data[hue] if hue else None

                style_by = st.selectbox(
                    "Style by",
                    options=color_by_opts,
                    key=PlotCustomiserStateKeys.StyleBy + plot_type,
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
                    key=PlotCustomiserStateKeys.CMap + plot_type,
                    help="Select a color map for the points in the plot.",
                )
                if cmap == "Custom":
                    cmap = []
                    for cat in config["hue"].unique():
                        color = st.color_picker(
                            f"Color for {cat}",
                            value="#1f77b4",
                            key=f"color_{cat}" + plot_type,
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
                    key="point_color" + plot_type,
                    help="Select a color for the points in the plot.",
                )

        elif plot_type == Plots.ConfusionMatrix.value:
            config["cmap"] = st.selectbox(
                "Color Map",
                options=["Blues", "Greens", "Oranges"],
                key=PlotCustomiserStateKeys.CMap + plot_type,
            )

            config["label_fontsize"] = st.slider(
                "Label Font Size",
                min_value=8,
                max_value=30,
                value=16,
                step=1,
                key=PlotCustomiserStateKeys.PlotLabelFontSize + plot_type,
                help="Select the font size for the labels in the confusion matrix.",
            )

            labels = data_options.class_names.keys()
            config["display_labels"] = []
            for label in labels:
                name = st.text_input(
                    f"Name for label for {label}",
                    value=data_options.class_names[label],
                    key=f"{PlotCustomiserStateKeys.LabelNames}_{label}" + plot_type,
                    help=f"Enter the label for the class '{label}'.",
                )
                config["display_labels"].append(name)

        elif plot_type == Plots.MatrixPlot:

            config["mask_upper_triangle"] = st.checkbox("Mask upper triangle")

            col1, col2 = st.columns(2)

            with col1:
                config["non_signifficant_colour"] = st.color_picker(
                    "Non-signifficant colour", value="#4575B4", key="non_signifficant" + plot_type
                )
            with col2:
                config["signifficant_colour"] = st.color_picker(
                    "Non-signifficant colour", value="#D73027", key="signifficant" + plot_type
                )

        elif plot_type == Plots.MetricsBoxPlot:

            cols = st.columns(3)

            with cols[0]:
                fill_colour = st.color_picker(
                    "Box Fill Colour",
                    value="#9aa0a5",
                    key="box_fill" + plot_type,
                    help="Select a fill color for the box in the box plot.",
                )
            with cols[1]:
                border_colour = st.color_picker(
                    "Box Border Colour",
                    value="#000000",
                    key="box_border" + plot_type,
                    help="Select a border color for the box in the box plot.",
                )
            with cols[2]:
                median_colour = st.color_picker(
                    "Median Line Colour",
                    value="#FF0000",
                    key="median_color" + plot_type,
                    help="Select a color for the median line in the box plot.",
                )

            config["fill_colour"] = fill_colour
            config["border_colour"] = border_colour
            config["median_colour"] = median_colour

        config["dpi"] = st.slider(
            "DPI",
            min_value=100,
            max_value=600,
            value=300,
            step=10,
            key="dpi" + plot_type,
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
