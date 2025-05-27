import streamlit as st

from polynet.app.options.plot import PlottingOptions
from polynet.app.options.state_keys import PlotOptionsStateKeys


def get_plotting_options() -> PlottingOptions:
    st.markdown("### 📊 Plotting Options")

    plot_axis_font_size = st.number_input(
        "Axis Font Size",
        min_value=6,
        max_value=40,
        value=st.session_state.get(PlotOptionsStateKeys.PlotAxisFontSize, 12),
        key=PlotOptionsStateKeys.PlotAxisFontSize,
    )
    plot_axis_tick_size = st.number_input(
        "Axis Tick Size",
        min_value=6,
        max_value=40,
        value=st.session_state.get(PlotOptionsStateKeys.PlotAxisTickSize, 10),
        key=PlotOptionsStateKeys.PlotAxisTickSize,
    )
    plot_colour_scheme = st.selectbox(
        "Colour Scheme",
        options=["Blues", "Greens", "Reds", "Purples", "Greys"],
        index=0,
        key=PlotOptionsStateKeys.PlotColourScheme,
    )
    dpi = st.number_input(
        "DPI",
        min_value=50,
        max_value=600,
        value=st.session_state.get(PlotOptionsStateKeys.DPI, 300),
        key=PlotOptionsStateKeys.DPI,
    )
    angle_rotate_xaxis_labels = st.slider(
        "Rotate X-axis Labels",
        min_value=0,
        max_value=90,
        value=st.session_state.get(PlotOptionsStateKeys.AngleRotateXaxisLabels, 0),
        key=PlotOptionsStateKeys.AngleRotateXaxisLabels,
    )
    angle_rotate_yaxis_labels = st.slider(
        "Rotate Y-axis Labels",
        min_value=0,
        max_value=90,
        value=st.session_state.get(PlotOptionsStateKeys.AngleRotateYaxisLabels, 0),
        key=PlotOptionsStateKeys.AngleRotateYaxisLabels,
    )
    save_plots = st.checkbox(
        "Save Plots",
        value=st.session_state.get(PlotOptionsStateKeys.SavePlots, True),
        key=PlotOptionsStateKeys.SavePlots,
    )
    plot_title_font_size = st.number_input(
        "Title Font Size",
        min_value=8,
        max_value=40,
        value=st.session_state.get(PlotOptionsStateKeys.PlotTitleFontSize, 14),
        key=PlotOptionsStateKeys.PlotTitleFontSize,
    )
    plot_font_family = st.selectbox(
        "Font Family",
        options=["sans-serif", "serif", "monospace", "Arial", "Times New Roman"],
        index=0,
        key=PlotOptionsStateKeys.PlotFontFamily,
    )
    height = st.number_input(
        "Plot Height (inches)",
        min_value=2.0,
        max_value=20.0,
        value=st.session_state.get(PlotOptionsStateKeys.Height, 6.0),
        key=PlotOptionsStateKeys.Height,
    )
    width = st.number_input(
        "Plot Width (inches)",
        min_value=2.0,
        max_value=20.0,
        value=st.session_state.get(PlotOptionsStateKeys.Width, 8.0),
        key=PlotOptionsStateKeys.Width,
    )
    plot_colour_map = st.selectbox(
        "Colour Map",
        options=["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm"],
        index=0,
    )

    return PlottingOptions(
        plot_axis_font_size=plot_axis_font_size,
        plot_axis_tick_size=plot_axis_tick_size,
        plot_colour_scheme=plot_colour_scheme,
        dpi=dpi,
        angle_rotate_xaxis_labels=angle_rotate_xaxis_labels,
        angle_rotate_yaxis_labels=angle_rotate_yaxis_labels,
        save_plots=save_plots,
        plot_title_font_size=plot_title_font_size,
        plot_font_family=plot_font_family,
        height=height,
        width=width,
        plot_colour_map=plot_colour_map,
    )
