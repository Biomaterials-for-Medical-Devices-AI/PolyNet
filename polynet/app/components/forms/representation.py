import streamlit as st
from rdkit.Chem import Descriptors
from polynet.app.options.state_keys import DescriptorCalculationStateKeys
import pandas as pd
from polynet.app.options.data import DataOptions
from polynet.app.options.allowable_sets import atom_properties, bond_features
from polynet.options.enums import AtomBondDescriptorDictKeys, AtomFeatures, BondFeatures


def get_descriptors_from_df(df: pd.DataFrame, data_options: DataOptions) -> list[str]:

    target_col = data_options.target_variable_col
    smiles_cols = data_options.smiles_cols
    id_col = data_options.id_col

    potential_descriptors = df.drop(
        columns=[target_col] + smiles_cols + [id_col], errors="ignore"
    ).columns.tolist()

    with st.expander("Descriptors from DataFrame", expanded=False):
        st.markdown(
            """
            The following columns were found in the dataset that could be used as potential descriptors.
            You can select which ones to use for the analysis.
            The descriptors selected here can be used either as independent representations of the molecules or in combination with the RDKit molecular descriptors.

            """
        )

        if not potential_descriptors:
            st.warning(
                "No descriptors found in the DataFrame. Please ensure that the DataFrame contains columns with molecular descriptors."
            )
            return []

        else:
            st.write(
                "The following columns are available as potential descriptors for your analysis:"
            )
            st.markdown("### Available Descriptors")
            descriptors_df = st.multiselect(
                "Select descriptors to use:",
                options=potential_descriptors,
                default=None,  # Preselect first 5 descriptors
                key=DescriptorCalculationStateKeys.DescriptorsDF,
                help="Choose which molecular descriptors from your data to use for the analysis.",
            )

            if descriptors_df:
                for desc in descriptors_df:
                    if not pd.api.types.is_numeric_dtype(df[desc]):
                        st.warning(
                            f"The column '{desc}' is not numeric and will be skipped. To use it as a descriptor, please ensure it contains numeric values."
                        )
                        descriptors_df.remove(desc)

            return descriptors_df


def molecular_descriptor_representation():
    with st.expander("Molecular Descriptors", expanded=False):

        all_descriptors = sorted([name for name, _ in Descriptors.descList])

        st.markdown("### Molecular Descriptor Selection")

        select_all = st.checkbox("Select all descriptors")

        if select_all:
            selected_descriptors = all_descriptors

        else:
            selected_descriptors = st.multiselect(
                "Select descriptors to calculate:",
                options=all_descriptors,
                default=["MolWt", "TPSA"],  # Preselect common ones
                key=DescriptorCalculationStateKeys.DescriptorsRDKit,
                help="Choose which RDKit descriptors to compute for the molecules.",
            )

        return selected_descriptors


def graph_representation():

    atomic_properties = sorted(atom_properties.keys())
    bond_properties = sorted(bond_features.keys())

    node_feats_config = {}
    edge_feats_config = {}

    with st.expander("Graph Representation", expanded=False):

        st.markdown("### Node and Edge Features Selection")

        st.markdown(
            """
            The graph representation is built using the RDKit library, which is a collection of cheminformatics and machine learning tools. For this representation, the SMILES strings are converted into a graph representation, where the atoms are the nodes and the bonds are the edges. This representation is used to build the graph neural networks (GNNs) that will be trained on your dataset.
            """
        )

        st.markdown("#### Atomic Properties")

        select_all_atomic = st.checkbox("Select all atomic properties")

        if select_all_atomic:
            selected_atomic_properties = atomic_properties
        else:
            selected_atomic_properties = st.multiselect(
                "Select atomic properties to compute:",
                options=atomic_properties,
                default=[
                    AtomFeatures.GetAtomicNum,
                    AtomFeatures.GetIsAromatic,
                    AtomFeatures.GetTotalDegree,
                    AtomFeatures.GetImplicitValence,
                ],
                key=DescriptorCalculationStateKeys.AtomProperties,
                help="Choose which RDKit Atom-level properties to extract.",
            )

        if selected_atomic_properties:
            total_num_node_feats = 0
            for prop in selected_atomic_properties:
                node_feats_config[prop] = {}
                if atom_properties[prop]:

                    selected_allowed_atom_properties = st.segmented_control(
                        label=f"Select allowable values for the property {prop}",
                        options=atom_properties[prop][AtomBondDescriptorDictKeys.Options],
                        default=atom_properties[prop].get(AtomBondDescriptorDictKeys.Default, []),
                        selection_mode="multi",
                        key=f"{DescriptorCalculationStateKeys.AtomProperties}_{prop}",
                        help=f"Choose whether to include the {prop} property in the graph representation.",
                    )

                    node_feats_config[prop][
                        AtomBondDescriptorDictKeys.AllowableVals
                    ] = selected_allowed_atom_properties

                    total_num_node_feats += len(selected_allowed_atom_properties)

                    node_feats_config[prop][AtomBondDescriptorDictKeys.Wildcard] = st.toggle(
                        label=f"Include wildcard placeholders for {prop}",
                        key=f"{DescriptorCalculationStateKeys.AtomProperties}_wildcard_{prop}",
                        help=f"Toggle to include wildcard placeholders for the {prop} property in the graph representation. This effectively encodes all none allowable values as a single placeholder, which can be useful for certain types of analyses.",
                    )
                    if node_feats_config[prop][AtomBondDescriptorDictKeys.Wildcard]:
                        total_num_node_feats += 1
                else:
                    total_num_node_feats += (
                        1  # For properties without options, count as one feature
                    )

            st.markdown(f"Total number of node features selected: {total_num_node_feats}")

        st.markdown("#### Bond Properties")

        select_all_bond = st.checkbox("Select all bond properties")

        if select_all_bond:
            selected_bond_properties = bond_properties
        else:
            selected_bond_properties = st.multiselect(
                "Select bond properties to compute:",
                options=bond_properties,
                default=[
                    BondFeatures.GetBondTypeAsDouble,
                    BondFeatures.GetIsAromatic,
                    BondFeatures.GetStereo,
                ],
                key=DescriptorCalculationStateKeys.BondProperties,
                help="Choose which RDKit Bond-level properties to extract.",
            )

        if selected_bond_properties:
            total_num_edge_feats = 0
            for prop in selected_bond_properties:
                edge_feats_config[prop] = {}
                if bond_features[prop]:

                    selected_allowed_bond_properties = st.segmented_control(
                        label=f"Select allowable values for the property {prop}",
                        options=bond_features[prop][AtomBondDescriptorDictKeys.Options],
                        default=bond_features[prop].get(AtomBondDescriptorDictKeys.Default, []),
                        selection_mode="multi",
                        key=f"{DescriptorCalculationStateKeys.BondProperties}_{prop}",
                        help=f"Choose whether to include the {prop} property in the graph representation.",
                    )

                    edge_feats_config[prop][
                        AtomBondDescriptorDictKeys.AllowableVals
                    ] = selected_allowed_bond_properties

                    total_num_edge_feats += len(selected_allowed_bond_properties)

                    edge_feats_config[prop][AtomBondDescriptorDictKeys.Wildcard] = st.toggle(
                        label="Include wildcard placeholders for this property",
                        key=f"{DescriptorCalculationStateKeys.BondProperties}_wildcard_{prop}",
                        help=f"Toggle to include wildcard placeholders for the {prop} property in the graph representation. This effectively encodes all none allowable values as a single placeholder, which can be useful for certain types of analyses.",
                    )

                    if edge_feats_config[prop][AtomBondDescriptorDictKeys.Wildcard]:
                        total_num_edge_feats += 1

                else:
                    total_num_edge_feats += (
                        1  # For properties without options, count as one feature
                    )

            st.markdown(f"Total number of edge features selected: {total_num_edge_feats}")

        return node_feats_config, edge_feats_config
