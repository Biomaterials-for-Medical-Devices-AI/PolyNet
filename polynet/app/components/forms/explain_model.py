from pathlib import Path

import pandas as pd
from rdkit.Chem import Descriptors
import streamlit as st
from torch_geometric.data import Dataset

from polynet.app.options.state_keys import ExplainModelStateKeys, ProjectionPlotStateKeys
from polynet.app.services.explain_model import (
    analyse_graph_embeddings,
    explain_model_global,
    explain_model_local,
)
from polynet.app.services.model_training import load_gnn_model
from polynet.app.utils import extract_number
from polynet.config.column_names import get_predicted_label_column_name, get_true_label_column_name
from polynet.config.constants import DataSet, ResultColumn
from polynet.config.enums import (
    AttributionPlotType,
    DescriptorMergingMethod,
    DimensionalityReduction,
    ExplainAlgorithm,
    FragmentationMethod,
    ImportanceNormalisationMethod,
    ProblemType,
)
from polynet.config.schemas import DataConfig
from polynet.featurizer import compute_rdkit_descriptors

# ---------------------------------------------------------------------------
# Shared molecule-set selector widget
# ---------------------------------------------------------------------------


def explain_mols_widget(
    data: pd.DataFrame, SetStateKey: str, ManuallySelectStateKey: str, MolsStateKey: str
) -> list:
    """Pills + optional manual multiselect for choosing a molecule set."""
    mols_to_explain = st.pills(
        "Select a set to explain",
        options=[DataSet.Training, DataSet.Validation, DataSet.Test, "All"],
        key=SetStateKey,
        default=["All"],
    )

    if mols_to_explain == DataSet.Training:
        explain_mols = data.loc[data[ResultColumn.SET] == DataSet.Training].index
    elif mols_to_explain == DataSet.Validation:
        explain_mols = data.loc[data[ResultColumn.SET] == DataSet.Validation].index
    elif mols_to_explain == DataSet.Test:
        explain_mols = data.loc[data[ResultColumn.SET] == DataSet.Test].index
    elif mols_to_explain == "All":
        explain_mols = data.index
    else:
        explain_mols = None

    if not mols_to_explain or st.checkbox("Manually refine selection", key=ManuallySelectStateKey):
        mols_to_plot = st.multiselect(
            "Select molecules", options=data.index, default=explain_mols, key=MolsStateKey
        )
    else:
        mols_to_plot = explain_mols.tolist()

    if not bool(mols_to_plot):
        st.error("Please select at least one molecule.")
        st.stop()

    return mols_to_plot


# ---------------------------------------------------------------------------
# Graph embedding projection
# ---------------------------------------------------------------------------


def embedding_projection(
    gnn_models: list,
    gnn_models_dir: Path,
    iterator_col: str,
    data: pd.DataFrame,
    preds: pd.DataFrame,
    dataset: Dataset,
    data_options: DataConfig,
    random_seed: int,
    weights_col: dict,
    merging_approach: DescriptorMergingMethod,
):
    if not st.checkbox(
        "Plot projection of graph embeddings",
        value=True,
        key=ProjectionPlotStateKeys.CreateProjectionPlot,
    ):
        return

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

    model_path = gnn_models_dir / model_name
    model = load_gnn_model(model_path)
    st.info(f"Model `{model_name}` loaded")

    predicted_col_name = get_predicted_label_column_name(
        target_variable_name=data_options.target_variable_name, model_name=model._name
    )
    iteration = extract_number(model_name)

    preds_projection = preds.loc[
        preds[iterator_col] == iteration, [ResultColumn.SET, predicted_col_name]
    ]
    projection_data = pd.merge(left=data, right=preds_projection, left_index=True, right_index=True)

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

    reduction_params["random_state"] = random_seed

    mols_to_plot = explain_mols_widget(
        data=projection_data,
        SetStateKey=ProjectionPlotStateKeys.PlotProjectionSet,
        ManuallySelectStateKey=ProjectionPlotStateKeys.ProjectionManualSelection,
        MolsStateKey=ProjectionPlotStateKeys.PlotProjectionMols,
    )

    colour_projection_by = st.selectbox(
        "Select a column to color the projection plot by",
        options=[data_options.target_variable_col] + [predicted_col_name] + ["Molecular property"],
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
            descriptor_df = compute_rdkit_descriptors(
                data=projection_data,
                smiles_cols=data_options.smiles_cols,
                descriptor_names=[descriptor],
                merging_approach=merging_approach,
                weights_col=weights_col,
            )
            projection_data = projection_data.join(descriptor_df)
            colour_projection_by = descriptor

    style_by = st.selectbox(
        "Select a column to style the projection plot by",
        options=[None, data_options.target_variable_col, predicted_col_name, ResultColumn.SET],
        key="style",
        index=3,
    )

    projection_data = projection_data.loc[mols_to_plot]
    labels = projection_data[colour_projection_by]
    style = projection_data[style_by] if style_by else None

    if st.checkbox("See data", key=ProjectionPlotStateKeys.ProjectionData):
        st.dataframe(projection_data)

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
            style_by=style,
            mols_to_plot=mols_to_plot,
            reduction_method=reduction_method,
            reduction_parameters=reduction_params,
            colormap=colour_map,
            label_name=colour_projection_by,
        )


# ---------------------------------------------------------------------------
# Shared parameters section (rendered above the explanation tabs)
# ---------------------------------------------------------------------------


def _parse_gnn_filename(fname: str) -> tuple[str, int]:
    """Return (architecture_name, iteration) from a GNN model filename like 'GCN_2.pt'."""
    stem = fname.rsplit(".", 1)[0]
    parts = stem.rsplit("_", 1)
    return parts[0], int(parts[1])


def _shared_params_section(
    gnn_models: list,
    gnn_models_dir: Path,
    iterator_col: str,
    data_options: DataConfig,
    preds: pd.DataFrame,
) -> dict | None:
    """
    Render the parameters shared by both explanation tabs and return them as a
    dict.  Returns ``None`` and stops rendering if no models are selected.
    """
    all_archs = sorted({_parse_gnn_filename(f)[0] for f in gnn_models})
    all_iterations = sorted({_parse_gnn_filename(f)[1] for f in gnn_models})

    select_all = st.toggle(
        "Select all GNN models", key=ExplainModelStateKeys.SelectAllModels, value=False
    )
    selected_archs = st.multiselect(
        "Select GNN architectures to explain",
        options=all_archs,
        default=all_archs if select_all else None,
        key=ExplainModelStateKeys.GNNArchSelector,
    )
    selected_iterations = st.multiselect(
        "Select bootstrap iterations",
        options=all_iterations,
        default=all_iterations,
        key=ExplainModelStateKeys.GNNBootstrapSelector,
        help="Each iteration is one train/test split. Explanations are averaged across selected iterations.",
    )

    if not selected_archs or not selected_iterations:
        st.info("Select at least one architecture and one bootstrap iteration.")
        return None

    selected_archs_set = set(selected_archs)
    selected_iterations_set = set(selected_iterations)
    selected_models = [
        f
        for f in gnn_models
        if _parse_gnn_filename(f)[0] in selected_archs_set
        and _parse_gnn_filename(f)[1] in selected_iterations_set
    ]

    models_dict = {}
    iterations = set()
    for model_file in selected_models:
        model_path = gnn_models_dir / model_file
        number = extract_number(model_file)
        gnn_model = load_gnn_model(model_path)
        gnn_model_name = gnn_model._name
        models_dict[f"{gnn_model_name}_{number}"] = gnn_model
        iterations.add(number)

    # Filter preds to the iterations present in the selected model set.
    # Deduplicate by index: when multiple models are selected, each mol_id
    # appears once per iteration; molecule selectors and label display need
    # exactly one row per molecule.
    _preds_iter = preds.loc[preds[iterator_col].isin(iterations)]
    preds_filtered = _preds_iter[~_preds_iter.index.duplicated(keep="first")]

    predicted_col_name = get_predicted_label_column_name(
        target_variable_name=data_options.target_variable_name, model_name=gnn_model_name
    )
    true_col_name = get_true_label_column_name(
        target_variable_name=data_options.target_variable_name
    )

    explain_algorithm = st.selectbox(
        "Explainability Algorithm",
        options=[ExplainAlgorithm.ChemistryMasking],
        key=ExplainModelStateKeys.ExplainAlgorithm,
    )

    fragmentation_approach = st.radio(
        "Fragmentation approach",
        options=[FragmentationMethod.BRICS, FragmentationMethod.MurckoScaffold],
        index=0,
        horizontal=True,
        key=ExplainModelStateKeys.FragmentationApproach,
    )

    # Target class (classification only)
    target_class = None
    if (
        explain_algorithm == ExplainAlgorithm.ChemistryMasking
        and data_options.problem_type == ProblemType.Classification
    ):
        if data_options.class_names:
            options = list(data_options.class_names.values())
        else:
            options = list(range(data_options.num_classes))

        target_class = st.selectbox(
            "Target class to explain",
            options=options,
            index=0,
            key=ExplainModelStateKeys.TargetClassSelector,
            help="The class whose predicted probability is used as the attribution signal (Y_pred − Y_pred_masked).",
        )
        if data_options.class_names:
            class_num = next(
                (k for k, v in data_options.class_names.items() if v == target_class), None
            )
            st.info(f"`{target_class}` corresponds to model output index `{class_num}`.")
            target_class = int(class_num)

    with st.expander("Advanced settings", expanded=False):
        normalisation_type = st.radio(
            "Normalisation",
            options=[
                ImportanceNormalisationMethod.Local,
                ImportanceNormalisationMethod.PerModel,
                ImportanceNormalisationMethod.Global,
                ImportanceNormalisationMethod.NoNormalisation,
            ],
            key=ExplainModelStateKeys.NormalisationMethodSelector,
            index=1,
            horizontal=True,
            help=(
                "**Local** — each (model × molecule) unit scaled to [−1, 1] independently.  \n"
                "**PerModel** — each model run scaled by its own maximum attribution.  \n"
                "**Global** — single divisor across all models and molecules.  \n"
                "**None** — raw scores."
            ),
        )

        cols = st.columns(2)
        with cols[0]:
            neg_color = st.color_picker(
                "Negative attribution colour",
                key=ExplainModelStateKeys.NegColorPlots,
                value="#40bcde",
            )
        with cols[1]:
            pos_color = st.color_picker(
                "Positive attribution colour",
                key=ExplainModelStateKeys.PosColorPlots,
                value="#e64747",
            )

    return {
        "models_dict": models_dict,
        "explain_algorithm": explain_algorithm,
        "fragmentation_approach": fragmentation_approach,
        "normalisation_type": normalisation_type,
        "target_class": target_class,
        "neg_color": neg_color,
        "pos_color": pos_color,
        "predicted_col_name": predicted_col_name,
        "true_col_name": true_col_name,
        "preds_filtered": preds_filtered,
    }


# ---------------------------------------------------------------------------
# Global explanation tab
# ---------------------------------------------------------------------------


def _global_tab_section(
    shared: dict,
    experiment_path: Path,
    dataset: Dataset,
    data: pd.DataFrame,
    data_options: DataConfig,
) -> None:
    st.markdown(
        "**Which molecular fragments drive predictions across the population?**  \n"
        "Select a set of molecules and run to see a distribution of fragment attributions "
        "across all selected models. Each data point in the ridge plot represents one "
        "real model prediction, preserving the full spread of the ensemble."
    )

    explain_mols = explain_mols_widget(
        data=shared["preds_filtered"],
        SetStateKey=ExplainModelStateKeys.GlobalExplainSet,
        ManuallySelectStateKey=ExplainModelStateKeys.GlobalExplainManuallySelector,
        MolsStateKey=ExplainModelStateKeys.GlobalExplainIDSelector,
    )

    cols = st.columns(2)
    with cols[0]:
        plot_type = st.radio(
            "Plot type",
            options=[AttributionPlotType.Ridge, AttributionPlotType.Bar, AttributionPlotType.Strip],
            format_func=lambda x: {
                AttributionPlotType.Ridge: "Ridge (full distribution)",
                AttributionPlotType.Bar: "Bar (mean ± 95 % CI)",
                AttributionPlotType.Strip: "Strip (individual scores + mean)",
            }[x],
            key=ExplainModelStateKeys.GlobalPlotType,
            horizontal=False,
        )
    with cols[1]:
        top_n = st.number_input(
            "Fragments to show (top N + bottom N)",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            key=ExplainModelStateKeys.TopNFragments,
            help="Show the N fragments with the highest and N with the lowest mean attribution.",
        )

    if st.button("Run Global Explanation") or st.toggle(
        "Keep running automatically", key=ExplainModelStateKeys.GlobalKeepRunning
    ):
        explain_model_global(
            models=shared["models_dict"],
            experiment_path=experiment_path,
            dataset=dataset,
            explain_mols=explain_mols,
            problem_type=data_options.problem_type,
            neg_color=shared["neg_color"],
            pos_color=shared["pos_color"],
            normalisation_type=shared["normalisation_type"],
            fragmentation_approach=shared["fragmentation_approach"],
            target_class=shared["target_class"],
            top_n=int(top_n),
            plot_type=plot_type,
        )


# ---------------------------------------------------------------------------
# Local explanation tab
# ---------------------------------------------------------------------------


def _local_tab_section(
    shared: dict,
    experiment_path: Path,
    dataset: Dataset,
    data: pd.DataFrame,
    data_options: DataConfig,
) -> None:
    st.markdown(
        "**How does a specific polymer's structure influence its prediction?**  \n"
        "Select one or more molecules to see per-atom attribution heatmaps and fragment "
        "importance tables. Attributions are averaged across the selected ensemble models."
    )

    local_mols = st.multiselect(
        "Select molecules to explain",
        options=sorted(shared["preds_filtered"].index.tolist()),
        key=ExplainModelStateKeys.LocalExplainIDSelector,
        default=None,
    )

    if not local_mols:
        st.info("Select at least one molecule above to generate local explanations.")
        return

    # Optional monomer name labels
    mol_names: dict = {}
    with st.expander("Set custom monomer names (optional)", expanded=False):
        for mol in local_mols:
            mol_names[mol] = []
            for smile_col in data_options.smiles_cols:
                name = st.text_input(
                    f"Name for {smile_col} in {mol}",
                    key=f"{ExplainModelStateKeys.LocalSetMolName}_{mol}_{smile_col}",
                )
                if name:
                    mol_names[mol].append(name)

    # Build predictions dict for true/pred label display
    preds_dict: dict = {}
    preds_filtered = shared["preds_filtered"]
    true_col = shared["true_col_name"]
    pred_col = shared["predicted_col_name"]
    class_names = data_options.class_names

    for mol in local_mols:
        preds_dict[mol] = {}
        try:
            if data_options.problem_type == ProblemType.Regression:
                vals = preds_filtered[[true_col, pred_col]].loc[mol].astype(str)
                preds_dict[mol][ResultColumn.PREDICTED] = vals.loc[pred_col]
                preds_dict[mol][ResultColumn.LABEL] = vals.loc[true_col]
            elif data_options.problem_type == ProblemType.Classification:
                vals = preds_filtered[[true_col, pred_col]].loc[mol].astype(int).astype(str)
                preds_dict[mol][ResultColumn.PREDICTED] = class_names[vals[pred_col]]
                preds_dict[mol][ResultColumn.LABEL] = class_names[vals[true_col]]
        except (KeyError, TypeError):
            pass  # predictions unavailable for this mol

    if st.button("Run Local Explanation") or st.toggle(
        "Keep running automatically", key=ExplainModelStateKeys.LocalKeepRunning
    ):
        explain_model_local(
            models=shared["models_dict"],
            experiment_path=experiment_path,
            dataset=dataset,
            explain_mols=local_mols,
            problem_type=data_options.problem_type,
            neg_color=shared["neg_color"],
            pos_color=shared["pos_color"],
            normalisation_type=shared["normalisation_type"],
            fragmentation_approach=shared["fragmentation_approach"],
            target_class=shared["target_class"],
            mol_names=mol_names,
            predictions=preds_dict,
            class_labels=data_options.class_names,
        )


# ---------------------------------------------------------------------------
# Main explanation form
# ---------------------------------------------------------------------------


def explain_predictions_form(
    experiment_path: Path,
    gnn_models: list,
    gnn_models_dir: Path,
    iterator_col: str,
    data_options: DataConfig,
    data: pd.DataFrame,
    preds: pd.DataFrame,
    dataset: Dataset,
) -> None:
    if not st.checkbox(
        "Explain model predictions", value=True, key=ExplainModelStateKeys.ExplainModels
    ):
        return
    # TODO: delete this and use explainability stage
    shared = _shared_params_section(
        gnn_models=gnn_models,
        gnn_models_dir=gnn_models_dir,
        iterator_col=iterator_col,
        data_options=data_options,
        preds=preds,
    )
    if shared is None:
        return

    tab_global, tab_local = st.tabs(
        ["Global Explanation — Population Trends", "Local Explanation — Single Molecule"]
    )

    with tab_global:
        _global_tab_section(
            shared=shared,
            experiment_path=experiment_path,
            dataset=dataset,
            data=data,
            data_options=data_options,
        )

    with tab_local:
        _local_tab_section(
            shared=shared,
            experiment_path=experiment_path,
            dataset=dataset,
            data=data,
            data_options=data_options,
        )
