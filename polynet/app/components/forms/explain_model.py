from pathlib import Path

import pandas as pd
from rdkit.Chem import Descriptors
import streamlit as st
from torch_geometric.data import Dataset

from polynet.app.options.data import DataOptions
from polynet.app.options.state_keys import ExplainModelStateKeys, ProjectionPlotStateKeys
from polynet.app.services.descriptors import (
    calculate_rdkit_df_dict,
    get_unique_smiles,
    merge_weighted,
)
from polynet.app.services.explain_model import analyse_graph_embeddings, explain_model
from polynet.app.services.model_training import load_gnn_model
from polynet.app.utils import (
    extract_number,
    get_predicted_label_column_name,
    get_true_label_column_name,
)
from polynet.options.enums import (
    DataSets,
    DimensionalityReduction,
    ExplainAlgorithms,
    ImportanceNormalisationMethods,
    ProblemTypes,
    Results,
)


def explain_mols_widget(
    data: pd.DataFrame, SetStateKey: str, ManuallySelectStateKey: str, MolsStateKey: str
):
    """
    Widget with logic to allow the user to explain just the selected molecules or set.
    """

    mols_to_explain = st.pills(
        "Select a set to explain",
        options=[DataSets.Training, DataSets.Validation, DataSets.Test, "All"],
        key=SetStateKey,
        default=["All"],
    )

    if mols_to_explain == DataSets.Training:
        explain_mols = data.loc[data[Results.Set.value] == DataSets.Training].index
    elif mols_to_explain == DataSets.Validation:
        explain_mols = data.loc[data[Results.Set.value] == DataSets.Validation].index
    elif mols_to_explain == DataSets.Test:
        explain_mols = data.loc[data[Results.Set.value] == DataSets.Test].index
    elif mols_to_explain == "All":
        explain_mols = data.index
    else:
        explain_mols = None

    if not mols_to_explain or st.checkbox(
        "Manually select points to show in projection plot", key=ManuallySelectStateKey
    ):
        mols_to_plot = st.multiselect(
            "Select the molecules you would like to see in the projection plot",
            options=data.index,
            default=explain_mols,
            key=MolsStateKey,
        )
    else:
        mols_to_plot = explain_mols.tolist()

    if not bool(mols_to_plot):
        st.error("Please select some datapoints to display on the plot.")
        st.stop()

    return mols_to_plot


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

        mols_to_plot = explain_mols_widget(
            data=projection_data,
            SetStateKey=ProjectionPlotStateKeys.PlotProjectionSet,
            ManuallySelectStateKey=ProjectionPlotStateKeys.ProjectionManualSelection,
            MolsStateKey=ProjectionPlotStateKeys.PlotProjectionMols,
        )

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


def explain_predictions_form(
    experiment_path: Path,
    gnn_models: list,
    gnn_models_dir: Path,
    iterator_col: str,
    data_options: DataOptions,
    data: pd.DataFrame,
    preds: pd.DataFrame,
    dataset: Dataset,
    node_feats: dict,
):
    if st.checkbox(
        "Explain model predictions", value=True, key=ExplainModelStateKeys.ExplainModels
    ):

        model = st.selectbox(
            "Select a GNN Model to Explain",
            options=sorted(gnn_models),
            key=ExplainModelStateKeys.ExplainModel,
        )

        if model:
            model_path = gnn_models_dir / model
            gnn_model = load_gnn_model(model_path)
            gnn_model_name = gnn_model._name
            number = extract_number(model)
            preds = preds.loc[preds[iterator_col] == number]

        else:
            st.error("Please select a GNN model to explain.")
            st.stop()

        predicted_col_name = get_predicted_label_column_name(
            target_variable_name=data_options.target_variable_name, model_name=gnn_model_name
        )

        true_col_name = get_true_label_column_name(
            target_variable_name=data_options.target_variable_name
        )

        explain_mol = explain_mols_widget(
            data=preds,
            SetStateKey=ExplainModelStateKeys.ExplainSet,
            ManuallySelectStateKey=ExplainModelStateKeys.ExplainManuallySelector,
            MolsStateKey=ExplainModelStateKeys.ExplainIDSelector,
        )

        plot_mols = st.multiselect(
            "Select Polymers ID to Plot",
            options=sorted(explain_mol),
            key=ExplainModelStateKeys.PlotIDSelector,
            default=None,
        )

        class_names = data_options.class_names

        preds_dict = {}
        if plot_mols:
            for mol in plot_mols:
                preds_dict[mol] = {}
                if data_options.problem_type == ProblemTypes.Regression:
                    predicted_vals = preds[[true_col_name, predicted_col_name]].loc[mol].astype(str)
                    preds_dict[mol][Results.Predicted] = predicted_vals.loc[predicted_col_name]
                    preds_dict[mol][Results.Label] = predicted_vals.loc[true_col_name]
                elif data_options.problem_type == ProblemTypes.Classification:
                    predicted_vals = (
                        preds[[true_col_name, predicted_col_name]].loc[mol].astype(int).astype(str)
                    )
                    preds_dict[mol][Results.Predicted] = class_names[
                        predicted_vals.loc[predicted_col_name]
                    ]
                    preds_dict[mol][Results.Label] = class_names[predicted_vals.loc[true_col_name]]

        mol_names = {}
        if st.toggle("Set Monomer Name"):

            for mol in explain_mol:
                mol_names[mol] = []
                for smile in data_options.smiles_cols:
                    name = st.text_input(
                        f"Enter a name for {smile} in {mol} for the plot",
                        key=f"{ExplainModelStateKeys.SetMolName}_{mol}_{smile}",
                    )
                    if name:
                        mol_names[mol].append(name)

        explain_algorithm = st.selectbox(
            "Select Explainability Algorithm",
            options=[
                ExplainAlgorithms.GNNExplainer,
                ExplainAlgorithms.IntegratedGradients,
                ExplainAlgorithms.Saliency,
                ExplainAlgorithms.InputXGradients,
                ExplainAlgorithms.Deconvolution,
                ExplainAlgorithms.ShapleyValueSampling,
                ExplainAlgorithms.GuidedBackprop,
            ],
            key=ExplainModelStateKeys.ExplainAlgorithm,
        )

        explain_feat = st.selectbox(
            "Select Node Features to Explain",
            options=["All Features"] + list(node_feats.keys()),
            index=0,
            key=ExplainModelStateKeys.ExplainNodeFeats,
        )

        cols = st.columns(2)
        with cols[0]:
            neg_color = st.color_picker(
                "Select Negative Color",
                key=ExplainModelStateKeys.NegColorPlots,
                value="#40bcde",  # Default red color
            )

        with cols[1]:
            pos_color = st.color_picker(
                "Select Positive Color",
                key=ExplainModelStateKeys.PosColorPlots,
                value="#e64747",  # Default green color
            )

        cutoff = st.select_slider(
            "Select the cutoff for explanations",
            options=[i / 10 for i in range(0, 11)],
            value=0.1,
            key=ExplainModelStateKeys.CutoffSelector,
        )

        normalisation_type = st.selectbox(
            "Select Normalisation Type",
            options=[
                ImportanceNormalisationMethods.Local,
                ImportanceNormalisationMethods.Global,
                ImportanceNormalisationMethods.NoNormalisation,
            ],
            key=ExplainModelStateKeys.NormalisationMethodSelector,
            index=1,
        )

        if st.button("Run Explanations") or st.toggle("Keep Running Explanations Automatically"):
            explain_model(
                model=gnn_model,
                model_number=number,
                experiment_path=experiment_path,
                dataset=dataset,
                explain_mols=explain_mol,
                plot_mols=plot_mols,
                explain_algorithm=explain_algorithm,
                problem_type=data_options.problem_type,
                neg_color=neg_color,
                pos_color=pos_color,
                cutoff_explain=cutoff,
                mol_names=mol_names,
                normalisation_type=normalisation_type,
                predictions=preds_dict,
                node_features=node_feats,
                explain_feature=explain_feat,
            )
