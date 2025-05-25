import streamlit as st
from rdkit.Chem import Descriptors
from polynet.app.options.state_keys import DescriptorCalculationStateKeys
import pandas as pd
from polynet.app.options.data import DataOptions
from polynet.app.options.allowable_sets import atom_properties, bond_features
from polynet.options.enums import (
    AtomBondDescriptorDictKeys,
    AtomFeatures,
    BondFeatures,
    DescriptorMergingMethods,
)
from polynet.app.utils import keep_only_numerical_columns, check_column_is_numeric


def merge_mols_approach(data_opts: DataOptions, df: pd.DataFrame) -> pd.DataFrame:

    with st.expander("Merge Molecules Approach", expanded=False):
        st.markdown(
            """
            When dealing with polymer datasets, it is often necessary to merge multiple SMILES strings into a single representation. This is particularly useful when the dataset contains polymers represented by multiple monomers.
            In this section, you can specify how to merge the numerical representations of the molecules.

            For molecular descriptors approaches, two strategies are available:
            1. **Average**: The numerical representations of the molecules are averaged across all monomers. This is useful when you want to capture the overall properties of the polymer.
            2. **Weighted Average**: representations of the molecules are multiplied by a weighting factor before adding the properties. This is useful when you want to give more importance to certain monomers in the polymer. The weighting factor can be specified in the dataset, and it is typically a column that contains the ratio of each monomer in the polymer.
            3. **Concatenation**: The numerical representations of the molecules are concatenated together. This method shall be used carefully, as it does not ensure permutation invariance, meaning that the order of the monomers matters.

            """
        )

        smiles_cols = data_opts.smiles_cols
        id_col = data_opts.id_col
        target_col = data_opts.target_variable_col

        potential_weighting_factors = df.drop(
            columns=[target_col] + smiles_cols + [id_col], errors="ignore"
        )

        potential_weighting_factors = keep_only_numerical_columns(
            df=potential_weighting_factors
        ).columns.tolist()

        if len(smiles_cols) >= 1:
            merging_methods = st.segmented_control(
                label="Select merging approach of descriptors for multiple SMILES columns",
                options=[
                    DescriptorMergingMethods.Average,
                    DescriptorMergingMethods.WeightedAverage,
                    DescriptorMergingMethods.Concatenate,
                ],
                selection_mode="multi",
                default="Average",  # Default to Average
                key=DescriptorCalculationStateKeys.MergeDescriptorsApproach,
                help="Choose how to merge the SMILES columns for the molecular representation. The selected approach will determine how the numerical representations of the molecules are combined.",
            )

            if not merging_methods:
                st.warning(
                    "Please select at least one merging approach for the SMILES columns to proceed."
                )

            if (
                DescriptorMergingMethods.WeightedAverage in merging_methods
                and len(potential_weighting_factors) > 0
            ):

                for col in smiles_cols:
                    weighting_factor = st.selectbox(
                        label=f"Select weighting factor for molecules in column {col}",
                        options=potential_weighting_factors,
                        key=f"{DescriptorCalculationStateKeys.WeightingFactor}_{col}",
                        help="Choose the column that contains the weighting factor for the SMILES column. This will be used to weight the numerical representations of the molecules.",
                    )

                    potential_weighting_factors.remove(weighting_factor)

            elif len(potential_weighting_factors) < len(smiles_cols):
                st.error(
                    "Not enough numerical columns to use as weighting factors for all the SMILES columns. Please ensure that the dataset contains enough numerical columns to use as weighting factors."
                )
                merging_methods.remove(DescriptorMergingMethods.WeightedAverage)

            if DescriptorMergingMethods.Concatenate in merging_methods:
                st.warning(
                    "Concatenation should only be used if the role of the molecules is different in the property to model, for example solute and solvent for solubility."
                )

            st.markdown(
                """
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

                for col in smiles_cols:

                    st.selectbox(
                        label=f"Select weighting factor for molecules in column {col}",
                        options=potential_weighting_factors,
                        key=f"{DescriptorCalculationStateKeys.WeightingFactor}_{col}",
                        help="Choose the column that contains the weighting factor for the SMILES column. This will be used to weight the numerical representations of the molecules.",
                    )

        else:
            st.write("Only one molecule column was indicated in the dataset, no merging is needed.")

    pass


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

                st.checkbox(
                    "Merge selected descriptors with RDKit descriptors",
                    value=True,
                    key=DescriptorCalculationStateKeys.MergeDescriptors,
                    help="If checked, the selected descriptors will be merged with the RDKit molecular descriptors for a comprehensive representation of the molecules.",
                )

                st.checkbox(
                    "Use selected descriptors as independent representations",
                    value=False,
                    key=DescriptorCalculationStateKeys.IndependentDFDescriptors,
                    help="If checked, the selected descriptors will be used as independent representations of the molecules.",
                )

            return descriptors_df


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

        st.markdown("### Molecular Descriptor Selection")

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

        if st.checkbox(
            "Use descriptors from the DataFrame",
            value=False,
            key=DescriptorCalculationStateKeys.UseDFDescriptors,
            help="If checked, the descriptors selected from the DataFrame will be used in addition to the RDKit descriptors.",
        ):

            if potential_descriptors:

                st.markdown("### Descriptors from DataFrame")

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
                st.markdown("### Available Descriptors")

                descriptors_df = st.multiselect(
                    "Select descriptors to use:",
                    options=potential_descriptors,
                    default=None,  # Preselect first 5 descriptors
                    key=DescriptorCalculationStateKeys.DescriptorsDF,
                    help="Choose which molecular descriptors from your data to use for the analysis.",
                )

                if not descriptors_df:
                    st.error("Please select at least one descriptor from the DataFrame to proceed.")

                else:

                    if selected_descriptors:

                        rdkit_independent = st.checkbox(
                            "Use RDkit Descriptors as Independent representations",
                            value=True,
                            key=DescriptorCalculationStateKeys.IndependentRDKitDescriptors,
                            help="If checked, the RDKit descriptors will be used as independent representations of the molecules.",
                        )

                        merge_rdkit_df = st.checkbox(
                            "Merge selected descriptors with RDKit descriptors",
                            value=True,
                            key=DescriptorCalculationStateKeys.MergeDescriptors,
                            help="If checked, the selected descriptors will be merged with the RDKit molecular descriptors for a comprehensive representation of the molecules.",
                        )

                        df_independent = st.checkbox(
                            "Use selected descriptors as independent representations",
                            value=True,
                            key=DescriptorCalculationStateKeys.IndependentDFDescriptors,
                            help="If checked, the selected descriptors will be used as independent representations of the molecules.",
                        )

                        if not rdkit_independent and not merge_rdkit_df and not df_independent:
                            st.warning(
                                "Please select at least one option to use the selected descriptors. "
                            )

                    potential_weighting_factors = [
                        column
                        for column in potential_weighting_factors
                        if column not in descriptors_df
                    ]

            else:
                st.warning(
                    "No numerical columns found in the DataFrame to use as descriptors. Please ensure that the DataFrame contains numeric columns that you'd like to use as descriptors."
                )

        else:
            descriptors_df = []
            if not selected_descriptors:
                st.warning("No descriptors have been selected. ")

        if len(smiles_cols) >= 1 and (selected_descriptors or descriptors_df):

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
                    DescriptorMergingMethods.Average,
                    DescriptorMergingMethods.WeightedAverage,
                    DescriptorMergingMethods.Concatenate,
                ],
                selection_mode="multi",
                default="Average",  # Default to Average
                key=DescriptorCalculationStateKeys.MergeDescriptorsApproach,
                help="Choose how to merge the SMILES columns for the molecular representation. The selected approach will determine how the numerical representations of the molecules are combined.",
            )

            if not merging_methods:
                st.warning(
                    "Please select at least one merging approach for the SMILES columns to proceed."
                )

            if (
                DescriptorMergingMethods.WeightedAverage in merging_methods
                and len(potential_weighting_factors) > 0
            ):

                for col in smiles_cols:
                    weighting_factor = st.selectbox(
                        label=f"Select weighting factor for molecules in column {col}",
                        options=potential_weighting_factors,
                        key=f"{DescriptorCalculationStateKeys.WeightingFactor}_{col}",
                        help="Choose the column that contains the weighting factor for the SMILES column. This will be used to weight the numerical representations of the molecules.",
                    )

                    potential_weighting_factors.remove(weighting_factor)

            elif len(potential_weighting_factors) < len(smiles_cols):
                st.error(
                    "Not enough numerical columns to use as weighting factors for all the SMILES columns. Please ensure that the dataset contains enough numerical columns to use as weighting factors."
                )
                merging_methods.remove(DescriptorMergingMethods.WeightedAverage)

            if DescriptorMergingMethods.Concatenate in merging_methods:
                st.warning(
                    "Concatenation should only be used if the role of the molecules is different in the property to model, for example solute and solvent for solubility."
                )


def graph_representation(data_opts: DataOptions, df: pd.DataFrame) -> tuple[dict, dict]:

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
                        st.selectbox(
                            label=f"Select weighting factor for molecules in column {col}",
                            options=potential_weighting_factors,
                            key=f"{DescriptorCalculationStateKeys.GraphWeightingFactor}_{col}",
                            help="Choose the column that contains the weighting factor for the SMILES column. This will be used to weight the numerical representations of the molecules.",
                        )
                else:
                    st.warning(
                        "No numerical columns found in the DataFrame to use as weighting factors. Please ensure that the DataFrame contains numeric columns that you'd like to use as weighting factors."
                    )

        return node_feats_config, edge_feats_config
