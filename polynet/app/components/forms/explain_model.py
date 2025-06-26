from pathlib import Path

import pandas as pd
from rdkit.Chem import Descriptors
import streamlit as st
from torch_geometric.data import Dataset

from polynet.app.options.data import DataOptions
from polynet.app.options.state_keys import ProjectionPlotStateKeys
from polynet.app.services.descriptors import (
    calculate_rdkit_df_dict,
    get_unique_smiles,
    merge_weighted,
)
from polynet.app.services.explain_model import analyse_graph_embeddings
from polynet.app.services.model_training import load_gnn_model
from polynet.app.utils import extract_number, get_predicted_label_column_name
from polynet.options.enums import DimensionalityReduction, Results, DataSets


def embedding_projection(
    gnn_models: list,
    gnn_models_dir: Path,
    iterator_col: str,
    data: pd.DataFrame,
    preds: pd.DataFrame,
    dataset: Dataset,
    data_options: DataOptions,
    random_seed: int,
    weights_col: dict,
):
    if st.checkbox(
        "Plot projection of graph embeddings",
        value=True,
        key=ProjectionPlotStateKeys.CreateProjectionPlot,
    ):

        reduction_params = {}

        # get the model to generate the embeddings from
        model_name = st.selectbox(
            "Select a GNN Model to get the embeddings from",
            options=sorted(gnn_models),
            key=ProjectionPlotStateKeys.ModelForProjections,
            index=0,
        )
        if not model_name:
            st.error("Please select a GNN model to plot.")
            st.stop()
        else:
            model_path = gnn_models_dir / model_name
            model = load_gnn_model(model_path)

        # get the name of the predicted col name for the model selected by the user
        predicted_col_name = get_predicted_label_column_name(
            target_variable_name=data_options.target_variable_name, model_name=model._name
        )
        # get the number of the iteration of the model
        iteration = extract_number(model_name)

        # filter the predictions to get only the predictions of the corresponding iteration and the corresponding model
        preds_projection = preds.loc[
            preds[iterator_col] == iteration, [Results.Set.value, predicted_col_name]
        ]
        projection_data = pd.concat([data, preds_projection], axis=1)

        # select the reduction method
        reduction_method = st.selectbox(
            "Select a reduction method",
            options=[DimensionalityReduction.tSNE, DimensionalityReduction.PCA],
            key=ProjectionPlotStateKeys.DimensionReductionMethod,
            index=0,
        )

        # set the perplexity in case that tSNE is selected
        if reduction_method == DimensionalityReduction.tSNE:
            perplexity = st.slider(
                "Select t-SNE perplexity",
                min_value=5,
                max_value=50,
                value=30,
                step=5,
                key=ProjectionPlotStateKeys.tSNEPerplexity,
            )
            reduction_params["perplexity"] = perplexity

        # set random seed for reproducibility
        reduction_params["random_state"] = random_seed

        # Select how to colour the points in the projection
        colour_projection_by = st.selectbox(
            "Select a column to color the projection plot by",
            options=[data_options.target_variable_col]
            + [predicted_col_name]
            + ["Molecular property"],
            key=ProjectionPlotStateKeys.ColourProjectionBy,
            index=0,
        )

        if colour_projection_by == "Molecular property":
            all_descriptors = sorted([name for name, _ in Descriptors.descList])
            descriptor = st.selectbox(
                "Select a descriptor to color the t-SNE plot by",
                options=all_descriptors,
                key=ProjectionPlotStateKeys.ProjectionDescriptorSelector,
                index=0,
            )

            if descriptor:
                unique_smiles = get_unique_smiles(projection_data, data_options.smiles_cols)
                descriptors = calculate_rdkit_df_dict(
                    unique_smiles, projection_data, data_options.smiles_cols, [descriptor]
                )
                projection_data = merge_weighted(
                    descriptors, projection_data, weights_col, projection_data
                )
                projection_data[descriptor] /= 100
                colour_projection_by = descriptor

        mols_for_projection = st.pills(
            "Select a set to explain",
            options=[DataSets.Training, DataSets.Validation, DataSets.Test, "All"],
            key=ProjectionPlotStateKeys.PlotProjectionSet,
            default=["All"],
        )

        if mols_for_projection == DataSets.Training:
            explain_mols = projection_data.loc[
                projection_data[Results.Set.value] == DataSets.Training
            ].index
        elif mols_for_projection == DataSets.Validation:
            explain_mols = projection_data.loc[
                projection_data[Results.Set.value] == DataSets.Validation
            ].index
        elif mols_for_projection == DataSets.Test:
            explain_mols = projection_data.loc[
                projection_data[Results.Set.value] == DataSets.Test
            ].index
        elif mols_for_projection == "All":
            explain_mols = projection_data.index
        else:
            explain_mols = None

        if not mols_for_projection or st.checkbox(
            "Manually select points to show in projection plot"
        ):
            mols_to_plot = st.multiselect(
                "Select the molecules you would like to see in the projection plot",
                options=projection_data.index,
                default=explain_mols,
                key=ProjectionPlotStateKeys.PlotProjectionMols,
            )
        else:
            mols_to_plot = explain_mols.tolist()

        if not bool(mols_to_plot):
            st.error("Please select some datapoints to display on the plot.")
            st.stop()

        projection_data = projection_data.loc[mols_to_plot]
        labels = projection_data[colour_projection_by]

        if st.checkbox("See data", key=ProjectionPlotStateKeys.ProjectionData):
            st.dataframe(projection_data)

        # select a colour map for the points
        colour_map = st.selectbox(
            "Select a colour map for the projection plot",
            options=["viridis", "plasma", "inferno", "magma", "cividis"],
            key=ProjectionPlotStateKeys.ProjectionColourMap,
            index=0,
        )

        if st.toggle("Plot Projection Plot", key=ProjectionPlotStateKeys.PlotProjection):

            analyse_graph_embeddings(
                model=model,
                dataset=dataset,
                labels=labels,
                mols_to_plot=mols_to_plot,
                reduction_method=reduction_method,
                reduction_parameters=reduction_params,
                colormap=colour_map,
            )
