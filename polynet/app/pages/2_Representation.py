from shutil import rmtree

import pandas as pd
import streamlit as st

from polynet.app.components.experiments import experiment_selector
from polynet.app.components.forms.representation import (
    graph_representation,
    molecular_descriptor_representation,
)
from polynet.app.options.data import DataOptions
from polynet.app.options.file_paths import (
    data_options_path,
    gnn_raw_data_file,
    gnn_raw_data_path,
    polynet_experiments_base_dir,
    representation_file,
    representation_file_path,
    representation_options_path,
    representation_parent_directory,
)
from polynet.app.options.representation import RepresentationOptions
from polynet.app.options.state_keys import DescriptorCalculationStateKeys
from polynet.app.services.configurations import load_options, save_options
from polynet.app.services.descriptors import build_vector_representation
from polynet.app.services.experiments import get_experiments
from polynet.app.services.plot import plot_molecule_3d
from polynet.app.utils import create_directory
from polynet.featurizer.graph_representation.polymer import CustomPolymerGraph
from polynet.options.enums import DescriptorMergingMethods


def parse_representation_options(
    experiment_name: str,
    node_feats: dict,
    edge_feats: dict,
    rdkit_descriptors: list,
    df_descriptors: list,
    data: pd.DataFrame,
    data_options: DataOptions,
    weights_col: dict,
):
    """
    Parse the representation options from the user input.
    This function is a placeholder for future implementation.
    """

    representation_options = RepresentationOptions(
        rdkit_descriptors=rdkit_descriptors,
        df_descriptors=df_descriptors,
        rdkit_independent=st.session_state.get(
            DescriptorCalculationStateKeys.IndependentRDKitDescriptors, bool(rdkit_descriptors)
        ),
        df_descriptors_independent=st.session_state.get(
            DescriptorCalculationStateKeys.IndependentDFDescriptors, bool(df_descriptors)
        ),
        mix_rdkit_df_descriptors=st.session_state.get(
            DescriptorCalculationStateKeys.MergeDescriptors, False
        ),
        smiles_merge_approach=st.session_state.get(
            DescriptorCalculationStateKeys.MergeDescriptorsApproach,
            DescriptorMergingMethods.NoMerging,
        ),
        node_feats=node_feats,
        edge_feats=edge_feats,
        weights_col=weights_col,
    )

    experiment_path = polynet_experiments_base_dir() / experiment_name

    representation_opts_path = representation_options_path(experiment_path=experiment_path)

    if representation_opts_path.exists():
        representation_opts_path.unlink()

    representation_path_dir = representation_parent_directory(experiment_path=experiment_path)
    if representation_path_dir.exists():
        rmtree(representation_path_dir)

    save_options(path=representation_opts_path, options=representation_options)

    if rdkit_descriptors or df_descriptors:

        descriptor_dfs = build_vector_representation(
            representation_opts=representation_options, data_options=data_options, data=data
        )

        create_directory(representation_file_path(experiment_path=experiment_path))

        for key, df in descriptor_dfs.items():
            if df is not None:
                df.to_csv(
                    representation_file(file_name=f"{key}.csv", experiment_path=experiment_path),
                    index=False,
                )

    if node_feats:

        data_for_GNN = data.copy()
        data_for_GNN = data_for_GNN[
            [data_options.id_col]
            + data_options.smiles_cols
            + list(weights_col.values())
            + [data_options.target_variable_col]
        ]

        create_directory(gnn_raw_data_path(experiment_path=experiment_path))

        # Save the raw data to a CSV file
        data_for_GNN.to_csv(
            gnn_raw_data_file(file_name=data_options.data_name, experiment_path=experiment_path),
            index=False,
        )

        dataset = CustomPolymerGraph(
            filename=data_options.data_name,
            root=gnn_raw_data_path(experiment_path=experiment_path).parent,
            smiles_cols=data_options.smiles_cols,
            target_col=data_options.target_variable_col,
            id_col=data_options.id_col,
            weights_col=weights_col,
            node_feats=node_feats,
            edge_feats=edge_feats,
        )

        st.write(dataset[0])

        if st.selectbox(
            "Select a molecule to display", options=data[data_options.id_col], key="mol_selection"
        ):
            if len(data_options.smiles_cols) > 1:
                mol = st.selectbox(
                    "Select which molecule to display",
                    options=data_options.smiles_cols,
                    key="smiles_selection",
                )
            else:
                mol = data_options.smiles_cols[0]
            smiles = data.loc[
                data[data_options.id_col] == st.session_state["mol_selection"], mol
            ].values[0]
            st.write(f"Displaying molecule: {smiles}")
            plot_molecule_3d(smiles=smiles)


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

    representation_opts = representation_options_path(experiment_path=experiment_path)

    if representation_opts.exists():
        st.error(
            "Representation configuration already exists for this experiment. "
            "You can modify the settings below, but be aware that this will overwrite the existing configuration."
        )

    path_to_data_opts = data_options_path(experiment_path=experiment_path)

    data_opts = load_options(path=path_to_data_opts, options_class=DataOptions)

    data = pd.read_csv(data_opts.data_path)

    st.write(data.describe())

    rdkit_descriptors, df_descriptors, descriptor_weights = molecular_descriptor_representation(
        df=data, data_options=data_opts
    )

    atomic_properties, bond_properties, graph_weights = graph_representation(
        data_opts=data_opts, df=data
    )

    if (descriptor_weights) and (graph_weights) and (not descriptor_weights == graph_weights):

        st.error(
            "The weights from the molecular descriptors and the weights from the graph representation are not equal. Please check your settings."
        )
        st.stop()

    if st.button("Apply Representation Settings"):
        parse_representation_options(
            experiment_name=experiment_name,
            node_feats=atomic_properties,
            edge_feats=bond_properties,
            rdkit_descriptors=rdkit_descriptors,
            df_descriptors=df_descriptors,
            data=data,
            data_options=data_opts,
            weights_col=graph_weights,
        )
