from shutil import rmtree
from typing import Dict, List

import pandas as pd
import streamlit as st

from polynet.app.components.experiments import experiment_selector
from polynet.app.components.forms.representation import (
    graph_representation,
    molecular_descriptor_representation,
    select_polymer_descriptors,
    select_weight_factor,
)
from polynet.app.options.file_paths import (
    data_options_path,
    polynet_experiment_path,
    polynet_experiments_base_dir,
    representation_options_path,
    representation_parent_directory,
)
from polynet.app.options.state_keys import DescriptorCalculationStateKeys
from polynet.app.services.configurations import load_options, save_options
from polynet.app.services.experiments import get_experiments
from polynet.app.services.plot import plot_molecule_3d
from polynet.config.enums import DescriptorMergingMethod, MolecularDescriptor
from polynet.config.schemas.data import DataConfig
from polynet.config.schemas.representation import RepresentationConfig
from polynet.pipeline import build_graph_dataset, compute_descriptors


def parse_representation_options(
    experiment_name: str,
    node_feats: Dict,
    edge_feats: Dict,
    molecular_descriptors: Dict[MolecularDescriptor, List[str]],
    data: pd.DataFrame,
    data_options: DataConfig,
    polymer_descriptors: List[str],
    weights_col: Dict[str, str],
):
    """
    Parse the representation options from the user input and run the
    corresponding pipeline stages (descriptor computation and/or graph
    dataset construction).
    """

    representation_options = RepresentationConfig(
        smiles_merge_approach=st.session_state.get(
            DescriptorCalculationStateKeys.MergeDescriptorsApproach,
            DescriptorMergingMethod.NoMerging,
        ),
        node_features=node_feats,
        edge_features=edge_feats,
        molecular_descriptors=molecular_descriptors,
        polymer_descriptors=polymer_descriptors,
        weights_col=weights_col,
    )

    experiment_path = polynet_experiment_path(experiment_name=experiment_name)

    representation_opts_path = representation_options_path(experiment_path=experiment_path)

    if representation_opts_path.exists():
        representation_opts_path.unlink()

    representation_path_dir = representation_parent_directory(experiment_path=experiment_path)
    if representation_path_dir.exists():
        rmtree(representation_path_dir)

    save_options(path=representation_opts_path, options=representation_options)

    if molecular_descriptors:
        # compute_descriptors saves CSVs with index (fixes the previous index=False
        # bug) and returns sanitised DataFrames ready for training.
        compute_descriptors(
            data=data,
            data_cfg=data_options,
            repr_cfg=representation_options,
            out_dir=experiment_path,
        )

    if node_feats:
        # build_graph_dataset filters the DataFrame, saves the raw GNN CSV and
        # constructs the CustomPolymerGraph, returning the dataset object.
        dataset = build_graph_dataset(
            data=data,
            data_cfg=data_options,
            repr_cfg=representation_options,
            out_dir=experiment_path,
        )

        # st.write(dataset[0])

        # if st.selectbox(
        #     "Select a molecule to display", options=data[data_options.id_col], key="mol_selection"
        # ):
        #     if len(data_options.smiles_cols) > 1:
        #         mol = st.selectbox(
        #             "Select which molecule to display",
        #             options=data_options.smiles_cols,
        #             key="smiles_selection",
        #         )
        #     else:
        #         mol = data_options.smiles_cols[0]
        #     smiles = data.loc[
        #         data[data_options.id_col] == st.session_state["mol_selection"], mol
        #     ].values[0]
        #     st.write(f"Displaying molecule: {smiles}")
        #     plot_molecule_3d(smiles=smiles)


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

    experiment_path = polynet_experiment_path(experiment_name=experiment_name)

    representation_opts = representation_options_path(experiment_path=experiment_path)

    if representation_opts.exists():
        st.error(
            "Representation configuration already exists for this experiment. "
            "You can modify the settings below, but be aware that this will overwrite the existing configuration."
        )

    path_to_data_opts = data_options_path(experiment_path=experiment_path)

    data_opts = load_options(path=path_to_data_opts, options_class=DataConfig)

    data = pd.read_csv(data_opts.data_path, index_col=0)

    st.write(data.describe())

    molecular_descriptors = molecular_descriptor_representation(df=data, data_options=data_opts)
    descriptors_weighted = (
        st.session_state.get(
            DescriptorCalculationStateKeys.MergeDescriptorsApproach,
            DescriptorMergingMethod.NoMerging,
        )
        == DescriptorMergingMethod.WeightedAverage
    )

    atomic_properties, bond_properties = graph_representation(data_opts=data_opts, df=data)
    graphs_weighted = st.session_state.get(
        DescriptorCalculationStateKeys.GraphWeightingFactor, False
    )

    requires_weights = descriptors_weighted or graphs_weighted

    weights = select_weight_factor(
        requires_weights=requires_weights, data_options=data_opts, df=data
    )

    polymer_descriptors = select_polymer_descriptors(
        data_options=data_opts, df=data, weight_cols=list(weights.values())
    )

    if requires_weights and not weights:
        disabled = True
        st.error(
            "To create the representation you require, we need the input of weighting columns for your molecules."
        )
    else:
        disabled = False

    if st.button("Apply Representation Settings", disabled=disabled):
        parse_representation_options(
            experiment_name=experiment_name,
            node_feats=atomic_properties,
            edge_feats=bond_properties,
            molecular_descriptors=molecular_descriptors,
            data=data,
            data_options=data_opts,
            polymer_descriptors=polymer_descriptors,
            weights_col=weights,
        )
