import streamlit as st
from polynet.app.services.experiments import get_experiments
from polynet.app.components.experiments import experiment_selector
from polynet.app.options.file_paths import polynet_experiments_base_dir, data_options_path
from pathlib import Path
from polynet.app.services.configurations import load_options
from polynet.app.options.data import DataOptions
import pandas as pd
from polynet.app.components.forms.representation import (
    merge_mols_approach,
    get_descriptors_from_df,
    molecular_descriptor_representation,
    graph_representation,
)
from polynet.featurizer.graph_representation.polymer import CustomPolymerGraph


def parse_representation_options():
    """
    Parse the representation options from the user input.
    This function is a placeholder for future implementation.
    """
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

    path_to_dataopts = data_options_path(
        experiment_path=polynet_experiments_base_dir() / experiment_name
    )

    data_opts = load_options(path=path_to_dataopts, options_class=DataOptions)

    data = pd.read_csv(data_opts.data_path)

    st.write(data.describe())

    # merge_mols_approach(data_opts=data_opts, df=data)

    molecular_descriptor_representation(df=data, data_options=data_opts)

    atomic_properties, bond_properties = graph_representation(data_opts=data_opts, df=data)

    # st.write(atomic_properties)
    # st.write(bond_properties)

    if st.button("Apply Representation Settings"):
        pass

    pass
