import pandas as pd
from rdkit.Chem import Descriptors
import streamlit as st

from polynet.app.options.allowable_sets import atom_properties, bond_features
from polynet.app.options.data import DataOptions
from polynet.app.options.state_keys import DescriptorCalculationStateKeys
from polynet.app.utils import keep_only_numerical_columns
from polynet.options.enums import (
    AtomBondDescriptorDictKeys,
    AtomFeatures,
    BondFeatures,
    DescriptorMergingMethods,
    MolecularDescriptors,
    StringRepresentation,
)
from polynet.utils.chem_utils import count_atom_property_frequency, count_bond_property_frequency


def select_weight_factor(def_value: bool, data_options: DataOptions, df: pd.DataFrame):

    mol_weights_col = {}

    if st.checkbox("Set molecular representation weighting factors", value=def_value):
        smiles_cols = data_options.smiles_cols
        target_col = data_options.target_variable_col
        id_col = data_options.id_col

        potential_weighting_factors = df.drop(
            columns=[target_col] + smiles_cols + [id_col], errors="ignore"
        )
        potential_weighting_factors = keep_only_numerical_columns(
            df=potential_weighting_factors
        ).columns.tolist()

        if potential_weighting_factors:

            for col in smiles_cols:
                weight_col = st.selectbox(
                    label=f"Select weighting factor for molecules in column {col}",
                    options=potential_weighting_factors,
                    key=f"{DescriptorCalculationStateKeys.WeightingFactor}_{col}",
                    index=None,
                    help="Choose the column that contains the weighting factor for the SMILES column. This will be used to weight the numerical representations of the molecules.",
                )

                if weight_col:
                    mol_weights_col[col] = weight_col
                    potential_weighting_factors.remove(weight_col)

        else:
            st.warning(
                "No numerical columns found in the DataFrame to use as weighting factors. Please ensure that the DataFrame contains numeric columns that you'd like to use as weighting factors."
            )

    return mol_weights_col


def molecular_descriptor_representation(df: pd.DataFrame, data_options: DataOptions):

    with st.expander("Molecular Descriptors", expanded=False):

        all_descriptors = sorted([name for name, _ in Descriptors.descList])

        smiles_cols = data_options.smiles_cols
        target_col = data_options.target_variable_col
        id_col = data_options.id_col

        potential_descriptors = df.drop(
            columns=[target_col] + smiles_cols + [id_col], errors="ignore"
        )
        potential_descriptors = keep_only_numerical_columns(
            df=potential_descriptors
        ).columns.tolist()

        potential_weighting_factors = potential_descriptors.copy()
        mol_weights_col = {}

        st.markdown("### Calculate RDKit Molecular Descriptors")

        select_all = st.checkbox(
            "Select all descriptors",
            value=False,
            key=DescriptorCalculationStateKeys.SelectAllRDKitDescriptors,
            help="If checked, all RDKit molecular descriptors will be selected by default. Uncheck to select specific descriptors.",
        )

        if select_all:
            default_descriptors = all_descriptors
        else:
            default_descriptors = [
                "MolWt",  # Molecular Weight
                "TPSA",  # Topological Polar Surface Area
                "NumHAcceptors",  # Number of Hydrogen Bond Acceptors
                "NumHDonors",  # Number of Hydrogen Bond Donors
                "MolLogP",  # Octanol-Water Partition Coefficient
            ]

        selected_descriptors = st.multiselect(
            "Select descriptors to calculate:",
            options=all_descriptors,
            default=default_descriptors,  # Preselect common ones
            key=DescriptorCalculationStateKeys.DescriptorsRDKit,
            help="Choose which RDKit descriptors to compute for the molecules.",
        )
        st.info(f"You have selected {len((selected_descriptors))} RDKit descriptors.")

        if data_options.string_representation == StringRepresentation.Smiles:
            disabled = True
        elif data_options.string_representation == StringRepresentation.PSmiles:
            disabled = False

        st.markdown("### polyBERT fingerprint")

        polybert_fps = st.checkbox(
            "Calculate polyBERT fingerprint",
            value=False,
            key=DescriptorCalculationStateKeys.polyBERTfp,
            disabled=True,
            help="polyBERT will remain disabled until Git repo becomes available again.",
        )

        st.markdown("### Descriptors from DataFrame")

        if potential_descriptors:

            st.markdown(
                """
                The following columns were found in the dataset that could be used as potential descriptors.
                You can select which ones to use for the analysis.
                The descriptors selected here can be used either as independent representations of the molecules or in combination with the RDKit molecular descriptors.

                """
            )

            st.write(
                "The following columns are available as potential descriptors for your analysis:"
            )

            descriptors_df = st.multiselect(
                "Select descriptors to use:",
                options=potential_descriptors,
                default=None,  # Preselect first 5 descriptors
                key=DescriptorCalculationStateKeys.DescriptorsDF,
                help="Choose which molecular descriptors from your data to use for the analysis.",
            )

            if descriptors_df:
                potential_weighting_factors = [
                    column for column in potential_weighting_factors if column not in descriptors_df
                ]

        else:
            st.warning(
                "No numerical columns found in the DataFrame to use as descriptors. Please ensure that the DataFrame contains numeric columns if you'd like to use as descriptors."
            )
            descriptors_df = []

        st.markdown("### Representation Options")

        if selected_descriptors:
            st.checkbox(
                "Use RDkit Descriptors as Independent representations",
                value=True,
                key=DescriptorCalculationStateKeys.IndependentRDKitDescriptors,
                help="If checked, the RDKit descriptors will be used as independent representations of the molecules.",
            )

        if descriptors_df:
            st.checkbox(
                "Use selected DF descriptors as independent representations",
                value=True,
                key=DescriptorCalculationStateKeys.IndependentDFDescriptors,
                help="If checked, the selected descriptors will be used as independent representations of the molecules.",
            )

        if selected_descriptors and descriptors_df:
            st.checkbox(
                "Merge selected descriptors with RDKit descriptors",
                value=True,
                key=DescriptorCalculationStateKeys.MergeDescriptors,
                help="If checked, the selected descriptors will be merged with the RDKit molecular descriptors for a comprehensive representation of the molecules.",
            )

        if polybert_fps:
            st.checkbox(
                "Use polyBERT fps as an independent representation",
                value=True,
                key=DescriptorCalculationStateKeys.polyBERTindependent,
                disabled=True,
                help="Currently, polyBERT fingerprints cannot be combined with any other representation.",
            )

        if len(smiles_cols) > 1 and (selected_descriptors or descriptors_df):

            st.markdown("### Merging Method for Multiple SMILES Columns")

            st.markdown(
                """
                When dealing with polymer datasets, it is often necessary to merge multiple SMILES strings into a single representation. This is particularly useful when the dataset contains polymers represented by multiple monomers.
                In this section, you can specify how to merge the numerical representations of the molecules.

                For molecular descriptor approaches, three strategies are available:
                1. **Average**: The numerical representations of the molecules are averaged across all molecules in each column. This is useful when you want to capture the overall properties of the polymer.
                2. **Weighted Average**: representations of the molecules are multiplied by a weighting factor before adding the properties. This is useful when you want to give more importance to certain monomers in the polymer. The weighting factor can be specified in the dataset, and it is typically a column that contains the ratio of each monomer in the polymer.
                3. **Concatenation**: The numerical representations of the molecules are concatenated together. This method shall be used carefully, as it does not ensure permutation invariance, meaning that the order of the monomers matters.

                """
            )

            merging_methods = st.segmented_control(
                label="Select merging approach of descriptors for multiple SMILES columns",
                options=[
                    # DescriptorMergingMethods.Average,
                    DescriptorMergingMethods.WeightedAverage,
                    DescriptorMergingMethods.Concatenate,
                ],
                default=DescriptorMergingMethods.WeightedAverage,
                key=DescriptorCalculationStateKeys.MergeDescriptorsApproach,
                help="Choose how to merge the SMILES columns for the molecular representation. The selected approach will determine how the numerical representations of the molecules are combined.",
            )

            if not merging_methods:
                st.error(
                    "Please select at least one merging approach for the SMILES columns to proceed."
                )
                st.stop()

            if DescriptorMergingMethods.WeightedAverage == merging_methods and len(
                potential_weighting_factors
            ) >= len(smiles_cols):

                for col in smiles_cols:
                    weight_col = st.selectbox(
                        label=f"Select weighting factor for molecules in column {col}",
                        options=potential_weighting_factors,
                        key=f"{DescriptorCalculationStateKeys.WeightingFactor}_{col}",
                        index=None,
                        help="Choose the column that contains the weighting factor for the SMILES column. This will be used to weight the numerical representations of the molecules.",
                    )

                    if weight_col:
                        mol_weights_col[col] = weight_col
                        potential_weighting_factors.remove(weight_col)

            elif len(potential_weighting_factors) < len(smiles_cols):
                st.error(
                    "Not enough numerical columns to use as weighting factors for all the SMILES columns. Please ensure that the dataset contains enough numerical columns to use as weighting factors."
                )
                merging_methods = None

            if not mol_weights_col and DescriptorMergingMethods.WeightedAverage == merging_methods:
                st.warning(
                    "No weighting factors selected for the SMILES columns. The numerical representations won't undergo weighted average."
                )
                merging_methods = None

            if DescriptorMergingMethods.Concatenate == merging_methods:
                st.warning(
                    "Concatenation should only be used if the role of the molecules is different in the property to model, for example solute and solvent for solubility."
                )
        else:
            mol_weights_col = {}

    descriptors_dict = {}
    descriptors_dict[MolecularDescriptors.RDKit] = selected_descriptors
    descriptors_dict[MolecularDescriptors.DataFrame] = descriptors_df

    return descriptors_dict, mol_weights_col


def graph_representation(data_opts: DataOptions, df: pd.DataFrame) -> tuple[dict, dict]:

    atomic_properties = sorted(atom_properties.keys())
    bond_properties = sorted(bond_features.keys())

    node_feats_config = {}
    edge_feats_config = {}
    mol_weights_col = {}

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
            default_atomic_properties = atomic_properties
        else:
            default_atomic_properties = [
                AtomFeatures.GetAtomicNum,
                AtomFeatures.GetTotalNumHs,
                AtomFeatures.GetTotalDegree,
                AtomFeatures.GetImplicitValence,
                AtomFeatures.GetIsAromatic,
            ]

        selected_atomic_properties = st.multiselect(
            "Select atomic properties to compute:",
            options=atomic_properties,
            default=default_atomic_properties,
            key=DescriptorCalculationStateKeys.AtomProperties,
            help="Choose which RDKit Atom-level properties to extract.",
        )

        if selected_atomic_properties:
            total_num_node_feats = 0

            for prop in selected_atomic_properties:

                prop_defaults = set()

                st.markdown(f"#### `{prop}` settings")

                show = st.checkbox(
                    f"Show frequencies of `{prop}` in the database",
                    key=f"{DescriptorCalculationStateKeys.ShowFrequency}_{prop}",
                )

                if show:
                    st.write(f"### Distribution of `{prop}`")

                cols = st.columns(2)

                for i, smiles_col in enumerate(data_opts.smiles_cols):

                    prop_counts = count_atom_property_frequency(df, smiles_col, prop)
                    prop_defaults.update(prop_counts.keys())

                    residue = int(i % 2)

                    with cols[residue]:

                        if show:
                            st.write(f"### Column `{smiles_col}`")
                            df_plot = pd.DataFrame.from_dict(
                                prop_counts, orient="index", columns=["Frequency"]
                            )
                            df_plot.index.name = str(prop)
                            df_plot = df_plot.sort_index()
                            st.bar_chart(df_plot, color="#ff3c00")

                node_feats_config[prop] = {}
                if atom_properties[prop]:

                    prop_defaults = [
                        val
                        for val in prop_defaults
                        if val in atom_properties[prop][AtomBondDescriptorDictKeys.Options]
                    ]

                    selected_allowed_atom_properties = st.segmented_control(
                        label=f"Select allowable values for the property `{prop}`",
                        options=atom_properties[prop][AtomBondDescriptorDictKeys.Options],
                        default=prop_defaults,
                        selection_mode="multi",
                        key=f"{DescriptorCalculationStateKeys.AtomProperties}_{prop}",
                        help=f"Choose whether to include the `{prop}` property in the graph representation.",
                    )

                    node_feats_config[prop][
                        AtomBondDescriptorDictKeys.AllowableVals
                    ] = selected_allowed_atom_properties

                    total_num_node_feats += len(selected_allowed_atom_properties)

                    node_feats_config[prop][AtomBondDescriptorDictKeys.Wildcard] = st.toggle(
                        label=f"Include wildcard placeholders for `{prop}`",
                        key=f"{DescriptorCalculationStateKeys.AtomProperties}_wildcard_{prop}",
                        help=f"Toggle to include wildcard placeholders for the `{prop}` property in the graph representation. This effectively encodes all none allowable values as a single placeholder, which can be useful for certain types of analyses.",
                    )
                    if node_feats_config[prop][AtomBondDescriptorDictKeys.Wildcard]:
                        total_num_node_feats += 1
                else:
                    st.write(
                        f"The possible values for the property `{prop}` are either `0` or `1` and cannot be changed. If you think this is an error, please raise an issue."
                    )
                    total_num_node_feats += (
                        1  # For properties without options, count as one feature
                    )

            st.markdown(f"**Total number of node features selected:** {total_num_node_feats}")

        st.markdown("#### Bond Properties")

        select_all_bond = st.toggle("Select all bond properties")

        if select_all_bond:
            default_bond_properties = bond_properties

        else:
            default_bond_properties = [
                BondFeatures.GetBondTypeAsDouble,
                BondFeatures.GetIsAromatic,
                BondFeatures.GetStereo,
            ]

        selected_bond_properties = st.multiselect(
            "Select bond properties to compute:",
            options=bond_properties,
            default=default_bond_properties,
            key=DescriptorCalculationStateKeys.BondProperties,
            help="Choose which RDKit Bond-level properties to extract.",
        )

        if selected_bond_properties:
            total_num_edge_feats = 0

            for prop in selected_bond_properties:

                prop_defaults = set()

                st.markdown(f"#### `{prop}` settings")

                show = st.checkbox(
                    f"Show frequencies of `{prop}` in the database",
                    key=f"{DescriptorCalculationStateKeys.ShowFrequency}_{prop}_bond",
                )

                if show:
                    st.write(f"### Distribution of `{prop}`")

                cols = st.columns(2)

                for i, smiles_col in enumerate(data_opts.smiles_cols):

                    prop_counts = count_bond_property_frequency(df, smiles_col, prop)
                    prop_defaults.update(prop_counts.keys())

                    residue = int(i % 2)

                    with cols[residue]:

                        if show:
                            st.write(f"### Column `{smiles_col}`")
                            df_plot = pd.DataFrame.from_dict(
                                prop_counts, orient="index", columns=["Frequency"]
                            )
                            df_plot.index.name = str(prop)
                            df_plot = df_plot.sort_index()
                            st.bar_chart(df_plot, color="#ff3c00")

                edge_feats_config[prop] = {}
                if bond_features[prop]:

                    prop_defaults = [
                        val
                        for val in prop_defaults
                        if val in bond_features[prop][AtomBondDescriptorDictKeys.Options]
                    ]

                    selected_allowed_bond_properties = st.segmented_control(
                        label=f"Select allowable values for the property `{prop}`",
                        options=bond_features[prop][AtomBondDescriptorDictKeys.Options],
                        default=prop_defaults,
                        selection_mode="multi",
                        key=f"{DescriptorCalculationStateKeys.BondProperties}_{prop}",
                        help=f"Choose whether to include the `{prop}` property in the graph representation.",
                    )

                    edge_feats_config[prop][
                        AtomBondDescriptorDictKeys.AllowableVals
                    ] = selected_allowed_bond_properties

                    total_num_edge_feats += len(selected_allowed_bond_properties)

                    edge_feats_config[prop][AtomBondDescriptorDictKeys.Wildcard] = st.toggle(
                        label=f"Include wildcard placeholders for `{prop}`",
                        key=f"{DescriptorCalculationStateKeys.BondProperties}_wildcard_{prop}",
                        help=f"Toggle to include wildcard placeholders for the `{prop}` property in the graph representation. This effectively encodes all none allowable values as a single placeholder, which can be useful for certain types of analyses.",
                    )

                    if edge_feats_config[prop][AtomBondDescriptorDictKeys.Wildcard]:
                        total_num_edge_feats += 1

                else:
                    st.write(
                        f"The possible values for the property `{prop}` are either `0` or `1` and cannot be changed. If you think this is an error, please raise an issue."
                    )
                    total_num_edge_feats += (
                        1  # For properties without options, count as one feature
                    )

            st.markdown(f"**Total number of edge features selected:** {total_num_edge_feats}")

        if not selected_atomic_properties and not selected_bond_properties:
            st.warning(
                "No atomic or bond properties selected. Please select at least one property to proceed with the graph representation."
            )
            return {}, {}

        smiles_cols = data_opts.smiles_cols

        if len(smiles_cols) > 1:
            target_col = data_opts.target_variable_col
            id_col = data_opts.id_col

            st.markdown(
                """
                When dealing with polymer datasets, it is often necessary to merge multiple SMILES strings into a single representation. This is particularly useful when the dataset contains polymers represented by multiple monomers.

                For graph representations, the merging approach is not applicable as each SMILES string is converted into a graph representation independently. The graph representation captures the structure of each molecule, and the graphs are automatically concatenated into a single graph by default.
                If you want to use a weighting factor for the graph representation, tick the box below.

                """
            )

            if st.checkbox(
                "Use weighting factors for graph representation",
                value=False,
                key=DescriptorCalculationStateKeys.GraphWeightingFactor,
                help="If checked, the numerical representations of the molecules will be weighted by a factor before merging them into a single graph representation.",
            ):

                potential_weighting_factors = df.drop(
                    columns=[target_col] + smiles_cols + [id_col], errors="ignore"
                )
                potential_weighting_factors = keep_only_numerical_columns(
                    df=potential_weighting_factors
                ).columns.tolist()

                if potential_weighting_factors:
                    for col in smiles_cols:
                        weight_col = st.selectbox(
                            label=f"Select weighting factor for molecules in column {col}",
                            options=potential_weighting_factors,
                            index=None,
                            key=f"{DescriptorCalculationStateKeys.GraphWeightingFactor}_{col}",
                            help="Choose the column that contains the weighting factor for the SMILES column. This will be used to weight the numerical representations of the molecules.",
                        )
                        if weight_col:
                            mol_weights_col[col] = weight_col
                            potential_weighting_factors.remove(weight_col)
                else:
                    st.warning(
                        "No numerical columns found in the DataFrame to use as weighting factors. Please ensure that the DataFrame contains numeric columns that you'd like to use as weighting factors."
                    )

        return node_feats_config, edge_feats_config, mol_weights_col
