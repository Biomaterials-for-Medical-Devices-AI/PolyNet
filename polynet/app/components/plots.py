import pandas as pd
import streamlit as st


def display_predictions(predictions_df: pd.DataFrame):
    """
    Display the predictions in the UI.

    Args:
        predictions_path (Path): The path to the predictions file.
    """
    st.write("### Predictions")
    st.write(predictions_df)


def display_plots(plots_path):

    for plot_file in plots_path.glob("*.png"):
        # Display each plot file in the directory
        st.image(plot_file, use_container_width=True)
