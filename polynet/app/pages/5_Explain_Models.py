import pandas as pd
import streamlit as st

from polynet.app.components.experiments import experiment_selector
from polynet.app.components.plots import display_plots, display_predictions, display_model_results
from polynet.app.options.data import DataOptions
from polynet.app.options.file_paths import (
    data_options_path,
    general_options_path,
    gnn_model_dir,
    gnn_plots_directory,
    gnn_raw_data_file,
    gnn_raw_data_path,
    ml_gnn_results_file_path,
    ml_results_parent_directory,
    polynet_experiments_base_dir,
    representation_file,
    representation_file_path,
    representation_options_path,
    train_gnn_model_options_path,
)
from polynet.app.options.representation import RepresentationOptions
from polynet.app.options.general_experiment import GeneralConfigOptions
from polynet.app.services.configurations import load_options
from polynet.app.services.experiments import get_experiments
from polynet.app.services.explain_model import explain_model
from polynet.app.services.model_training import load_gnn_model
from polynet.app.utils import filter_dataset_by_ids
from polynet.featurizer.graph_representation.polymer import CustomPolymerGraph
from polynet.options.enums import ExplainAlgorithms, DataSets, Results
from polynet.app.utils import extract_number, get_iterator_name, get_predicted_label_column_name


def run_explanations(
    dataset: CustomPolymerGraph, explain_mol: str, gnn_model, explain_algorithm: ExplainAlgorithms
):
    pass


st.header("Representation of Polymers")

st.markdown(
    """
    In this section, you can build the representation of your polymers. This representation will be the input for the ML models you will train. The representation is built using the SMILES strings of the monomers, which are the building blocks of the polymers.
    We currently support two types of representations:
    1. **Graph Representation**: This representation is built using the SMILES strings of the monomers. The graph representation is built using the RDKit library, which is a collection of cheminformatics and machine learning tools. For this representation, the SMILES strings are converted into a graph representation, where the atoms are the nodes and the bonds are the edges. This representation is used to build the graph neural networks (GNNs) that will be trained on your dataset.
    2. **Molecular Descriptors**: This representation is built using the RDKit library. The molecular descriptors are a set of numerical values that describe the structure and properties of the molecule. This approach effectively transforms the molecule into a vector representation. You can also use descriptors from the dataset, which you can concatenate with the RDkit descriptors or use them as a separate input to the model.
    """
)


choices = get_experiments()
experiment_name = experiment_selector(choices)


if experiment_name:

    experiment_path = polynet_experiments_base_dir() / experiment_name

    path_to_data_opts = data_options_path(
        experiment_path=polynet_experiments_base_dir() / experiment_name
    )

    data_options = load_options(path=path_to_data_opts, options_class=DataOptions)

    path_to_representation_opts = representation_options_path(
        experiment_path=polynet_experiments_base_dir() / experiment_name
    )
    representation_options = load_options(
        path=path_to_representation_opts, options_class=RepresentationOptions
    )

    train_gnn_options = train_gnn_model_options_path(
        experiment_path=polynet_experiments_base_dir() / experiment_name
    )

    path_to_general_opts = general_options_path(experiment_path=experiment_path)

    general_experiment_options = load_options(
        path=path_to_general_opts, options_class=GeneralConfigOptions
    )

    iterator_col = get_iterator_name(general_experiment_options.split_type)

    if not train_gnn_options.exists():
        st.error(
            "No models have been trained yet. Please train a model first in the 'Train GNN' section."
        )
        st.stop()

    display_model_results(experiment_path=experiment_path, expanded=False)

    dataset = CustomPolymerGraph(
        filename=data_options.data_name,
        root=gnn_raw_data_path(experiment_path=experiment_path).parent,
        smiles_cols=data_options.smiles_cols,
        target_col=data_options.target_variable_col,
        id_col=data_options.id_col,
        weights_col=representation_options.weights_col,
        node_feats=representation_options.node_feats,
        edge_feats=representation_options.edge_feats,
    )

    data = pd.read_csv(
        gnn_raw_data_file(file_name=data_options.data_name, experiment_path=experiment_path),
        index_col=0,
    )

    preds = pd.read_csv(
        ml_gnn_results_file_path(experiment_path=experiment_path, file_name="predictions.csv"),
        index_col=0,
    )

    gnn_models_dir = gnn_model_dir(experiment_path=experiment_path)

    gnn_models = [
        model.name
        for model in gnn_models_dir.iterdir()
        if model.is_file() and model.suffix == ".pt"
    ]

    model = st.selectbox(
        "Select a GNN Model to Explain", options=sorted(gnn_models), key="gnn_model_selector"
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

    set = st.pills(
        "Select a set to explain",
        options=[DataSets.Training, DataSets.Validation, DataSets.Test],
        key="set_selector",
    )

    if set == DataSets.Training:
        explain_mols = preds.loc[preds[Results.Set.value] == DataSets.Training, Results.Index.value]
    elif set == DataSets.Validation:
        explain_mols = preds.loc[
            preds[Results.Set.value] == DataSets.Validation, Results.Index.value
        ]
    elif set == DataSets.Test:
        explain_mols = preds.loc[preds[Results.Set.value] == DataSets.Test, Results.Index.value]
    else:
        explain_mols = preds.sample(5).index[:5]

    explain_mol = st.multiselect(
        "Select Polymers ID to Calculate Explaination",
        options=data.index,
        key="polymer_id_selector",
        default=explain_mols,
    )

    plot_mols = st.multiselect(
        "Select Polymers ID to Plot",
        options=explain_mol,
        key="polymer_id_plot_selector",
        default=explain_mol,
    )

    if plot_mols:
        preds_dict = {}
        pass

    mol_names = {}
    if st.toggle("Set Monomer Name"):

        for mol in explain_mol:
            mol_names[mol] = []
            for smile in data_options.smiles_cols:
                name = st.text_input(
                    f"Enter a name for {smile} in {mol} for the plot", key=f"mol_name_{mol}_{smile}"
                )
                if name:
                    mol_names[mol].append(name)

    explain_algorithm = st.selectbox(
        "Select Explainability Algorithm",
        options=[
            # ExplainAlgorithms.GNNExplainer,
            # ExplainAlgorithms.IntegratedGradients,
            # ExplainAlgorithms.Saliency,
            # ExplainAlgorithms.InputXGradients,
            # ExplainAlgorithms.Deconvolution,
            ExplainAlgorithms.ShapleyValueSampling,
            # ExplainAlgorithms.GuidedBackprop,
        ],
        key="explain_algorithm_selector",
    )

    cols = st.columns(2)
    with cols[0]:
        neg_color = st.color_picker(
            "Select Negative Color", key="neg_color_picker", value="#40bcde"  # Default red color
        )

    with cols[1]:
        pos_color = st.color_picker(
            "Select Positive Color", key="pos_color_picker", value="#e64747"  # Default green color
        )

    cutoff = st.select_slider(
        "Select the cutoff for explanations",
        options=[i / 10 for i in range(0, 11)],
        value=0.1,
        key="cutoff_selector",
    )

    normalisation_type = st.selectbox(
        "Select Normalisation Type", options=["local", "global"], key="normalisation_type_selector"
    )
    if normalisation_type == "global":
        st.selectbox(
            "Which molecules to consider for global normalisation",
            options=["All molecules", "Selected molecules only"],
            index=0,
            key="global_normalisation_selector",
        )

    if st.button("Run Explanations") or st.toggle("Keep Running Explanations Automatically"):
        explain_model(
            model=gnn_model,
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
        )
        st.write(f"Running {explain_algorithm} on Polymer ID: {explain_mol}...")
