from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from polynet.app.options.data import DataOptions
from polynet.app.options.file_paths import representation_file, representation_file_path
from polynet.app.options.general_experiment import GeneralConfigOptions
from polynet.app.options.representation import RepresentationOptions
from polynet.app.options.search_grids import get_grid_search
from polynet.app.options.train_TML import TrainTMLOptions
from polynet.options.enums import ProblemTypes, Results, TradtionalMLModels, TransformDescriptors


def train_tml_model(
    train_tml_options: TrainTMLOptions,
    tml_models: dict,
    general_experiment_options: GeneralConfigOptions,
    representation_options: RepresentationOptions,
    data_options: DataOptions,
    experiment_path: Path,
    train_val_test_idxs: tuple,
):

    dataframes = load_dataframes(
        representation_options=representation_options,
        experiment_path=experiment_path,
        target_variable_col=data_options.target_variable_col,
    )

    train_ids, val_ids, test_ids = train_val_test_idxs

    trained_models = {}
    training_data = {}
    scalers = {}

    for i, (train_idxs, val_idxs, test_idxs) in enumerate(zip(train_ids, val_ids, test_ids)):

        iteration = i + 1
        train_idxs += val_idxs

        for df_name, df in dataframes.items():

            train_df = df.loc[train_idxs]
            val_df = df.loc[val_idxs]
            test_df = df.loc[test_idxs]

            log_name = f"{df_name}_{iteration}"

            if train_tml_options.TransformFeatures != TransformDescriptors.NoTransformation:

                original_train_features = train_df.iloc[:, :-1].copy()

                train_features, scaler = transform_dependent_variables(
                    fit_data=original_train_features,
                    transform_data=train_df.iloc[:, :-1],
                    transform_type=train_tml_options.TransformFeatures,
                )
                train_df.iloc[:, :-1] = train_features

                val_features = scaler.transform(val_df.iloc[:, :-1])
                val_df.iloc[:, :-1] = val_features

                test_features = scaler.transform(test_df.iloc[:, :-1])
                test_df.iloc[:, :-1] = test_features

                scalers[log_name] = scaler

            training_data[log_name] = (train_df, val_df, test_df)

            models = generate_models(train_tml_options, data_options.problem_type)

            for model_name, model in models.items():

                model = get_model(
                    model_type=model,
                    model_params=tml_models[model_name],
                    random_state=general_experiment_options.random_seed + i,
                )

                if train_tml_options.HyperparameterOptimization:

                    param_grid = get_grid_search(
                        model_name=model_name,
                        random_seed=general_experiment_options.random_seed + i,
                        problem_type=data_options.problem_type,
                    )

                    if data_options.problem_type == ProblemTypes.Classification:
                        cv = StratifiedKFold(
                            n_splits=5,
                            shuffle=True,
                            random_state=general_experiment_options.random_seed + i,
                        )
                    else:
                        cv = 5

                    grid_search = GridSearchCV(
                        estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1
                    )

                    grid_search.fit(train_df.iloc[:, :-1], train_df.iloc[:, -1])
                    model = grid_search.best_estimator_
                    print("Best parameters found: ", grid_search.best_params_)
                    print(model)

                else:
                    print("Fitting model without hyperparameter optimization...")
                    model.fit(train_df.iloc[:, :-1], train_df.iloc[:, -1])

                model_log_name = f"{model_name.replace(' ' ,'')}-{log_name}"
                trained_models[model_log_name] = model

    return trained_models, training_data, scalers


def get_model(model_type: type, model_params: dict = None, random_state: int = None):
    """Get a new instance of the requested machine learning model.

    If the model is to be used in a grid search, specify `model_params=None`.

    Args:
        model_type (type): The Python type (constructor) of the model to instantiate.
        model_params (dict, optional): The parameters to pass to the model constructor. Defaults to None.

    Returns:
        MlModel: A new instance of the requested machine learning model.
    """

    if model_params is not None and model_type not in [LinearRegression]:
        model_params["random_state"] = random_state

    return model_type(**model_params) if model_params is not None else model_type()


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


def generate_models(train_tml_options: TrainTMLOptions, problem_type: ProblemTypes):

    models = {}

    if train_tml_options.TrainLinearRegression:
        models[TradtionalMLModels.LinearRegression] = LinearRegression

    if train_tml_options.TrainLogisticRegression:
        models[TradtionalMLModels.LogisticRegression] = LogisticRegression

    if train_tml_options.TrainRandomForest:
        if problem_type == ProblemTypes.Classification:
            models[TradtionalMLModels.RandomForest] = RandomForestClassifier
        elif problem_type == ProblemTypes.Regression:
            models[TradtionalMLModels.RandomForest] = RandomForestRegressor

    if train_tml_options.TrainSupportVectorMachine:
        if problem_type == ProblemTypes.Classification:
            models[TradtionalMLModels.SupportVectorMachine] = SVC
        elif problem_type == ProblemTypes.Regression:
            models[TradtionalMLModels.SupportVectorMachine] = SVR

    if train_tml_options.TrainXGBoost:
        if problem_type == ProblemTypes.Classification:
            models[TradtionalMLModels.XGBoost] = XGBClassifier
        elif problem_type == ProblemTypes.Regression:
            models[TradtionalMLModels.XGBoost] = XGBRegressor

    return models


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
