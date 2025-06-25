from pathlib import Path

import pandas as pd
from rdkit.Chem import Descriptors
import streamlit as st
from torch_geometric.data import Dataset

from polynet.app.options.data import DataOptions
from polynet.app.options.general_experiment import GeneralConfigOptions
from polynet.app.options.representation import RepresentationOptions
from polynet.app.options.state_keys import ProjectionPlotStateKeys
from polynet.app.services.descriptors import (
    calculate_rdkit_df_dict,
    get_unique_smiles,
    merge_weighted,
)
from polynet.app.services.explain_model import analyse_graph_embeddings
from polynet.app.services.model_training import load_gnn_model
from polynet.app.utils import extract_number, get_predicted_label_column_name
from polynet.options.enums import DimensionalityReduction, Results


def embedding_projection(
    gnn_models: list,
    gnn_models_dir: Path,
    iterator_col: str,
    data: pd.DataFrame,
    preds: pd.DataFrame,
    dataset: Dataset,
    data_options: DataOptions,
    general_experiment_options: GeneralConfigOptions,
    representation_options: RepresentationOptions,
):
    if st.checkbox(
        "Plot projection of graph embeddings",
        value=True,
        key=ProjectionPlotStateKeys.CreateProjectionPlot,
    ):

        reduction_params = {}

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

        predicted_col_name = get_predicted_label_column_name(
            target_variable_name=data_options.target_variable_name, model_name=model._name
        )
        iteration = extract_number(model_name)

        preds_projection = preds.loc[
            preds[iterator_col] == iteration, [Results.Set.value, predicted_col_name]
        ]
        projection_data = pd.concat([data, preds_projection], axis=1)

        reduction_method = st.selectbox(
            "Select a reduction method",
            options=[DimensionalityReduction.tSNE, DimensionalityReduction.PCA],
            key=ProjectionPlotStateKeys.DimensionReductionMethod,
            index=0,
        )

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

        reduction_params["random_state"] = general_experiment_options.random_seed

        color_tsne_by = st.selectbox(
            "Select a column to color the projection plot by",
            options=[data_options.target_variable_col] + [predicted_col_name] + ["Descriptors"],
            key=ProjectionPlotStateKeys.ColourProjectionBy,
            index=0,
        )

        colour_map = st.selectbox(
            "Select a colour map for the projection plot",
            options=["viridis", "plasma", "inferno", "magma", "cividis"],
            key=ProjectionPlotStateKeys.ProjectionColourMap,
            index=0,
        )

        if color_tsne_by == "Descriptors":
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
                    descriptors,
                    projection_data,
                    representation_options.weights_col,
                    projection_data,
                )
                projection_data[descriptor] /= 100
                color_tsne_by = descriptor
                st.write(projection_data)

        if st.toggle("Plot Projection Plot", key=ProjectionPlotStateKeys.PlotProjection):

            analyse_graph_embeddings(
                model=model,
                dataset=dataset,
                data=projection_data,
                reduction_method=reduction_method,
                reduction_parameters=reduction_params,
                colormap=colour_map,
                colour_by=color_tsne_by,
            )
