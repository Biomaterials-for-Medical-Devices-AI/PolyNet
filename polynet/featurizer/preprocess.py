from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from polynet.app.options.file_paths import representation_file
from polynet.options.enums import TransformDescriptors


def transform_dependent_variables(
    fit_data: pd.DataFrame, transform_data: pd.DataFrame, transform_type: TransformDescriptors
):
    """
    Transform the dependent variable based on the specified transformation type.
    """
    if transform_type == TransformDescriptors.StandardScaler:

        scaler = StandardScaler()
        scaler.fit(fit_data)
        transform_data = scaler.transform(transform_data)

    elif transform_type == TransformDescriptors.MinMaxScaler:

        scaler = MinMaxScaler()
        scaler.fit(fit_data)
        transform_data = scaler.transform(transform_data)

    else:
        raise ValueError(f"Unsupported transformation type: {transform_type}")

    return transform_data, scaler


def load_dataframes(
    representation_options: RepresentationOptions, experiment_path: Path, target_variable_col: str
):

    dataframe_dict = {}

    if representation_options.rdkit_independent:

        rdkit_file_path = representation_file(
            experiment_path=experiment_path, file_name="RDKit.csv"
        )

        rdkit_df = pd.read_csv(rdkit_file_path, index_col=0)

        rdkit_df = sanitise_df(
            df=rdkit_df,
            descriptors=representation_options.rdkit_descriptors,
            target_variable_col=target_variable_col,
        )

        dataframe_dict["RDKit"] = rdkit_df

    if representation_options.df_descriptors_independent:
        df_file_path = representation_file(experiment_path=experiment_path, file_name="DF.csv")

        df_df = pd.read_csv(df_file_path, index_col=0)

        df_df = sanitise_df(
            df=df_df,
            descriptors=representation_options.df_descriptors,
            target_variable_col=target_variable_col,
        )

        dataframe_dict["DF"] = df_df

    if representation_options.mix_rdkit_df_descriptors:

        mix_file_path = representation_file(
            experiment_path=experiment_path, file_name="RDKit_DF.csv"
        )

        mix_df = pd.read_csv(mix_file_path, index_col=0)

        mix_df = sanitise_df(
            df=mix_df,
            descriptors=representation_options.rdkit_descriptors
            + representation_options.df_descriptors,
            target_variable_col=target_variable_col,
        )

        dataframe_dict["RDKit_DF"] = mix_df

    if representation_options.polybert_fp:
        polybert_file_path = representation_file(
            experiment_path=experiment_path, file_name="polyBERT.csv"
        )
        polybert_df = pd.read_csv(polybert_file_path, index_col=0)
        polybert_df = sanitise_df(
            df=polybert_df,
            descriptors=[f"polyBERT_{i}" for i in range(600)],
            target_variable_col=target_variable_col,
        )
        dataframe_dict["polyBERT"] = polybert_df

    return dataframe_dict


def sanitise_df(df: pd.DataFrame, descriptors: list, target_variable_col: str):

    clean_df = df.copy()

    clean_df = clean_df[descriptors + [target_variable_col]].dropna(axis=1)

    return clean_df
