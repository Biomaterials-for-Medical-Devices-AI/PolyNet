from copy import deepcopy
from pathlib import Path

import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold


from polynet.app.options.data import DataOptions

from polynet.app.options.general_experiment import GeneralConfigOptions
from polynet.app.options.representation import RepresentationOptions
from polynet.options.search_grids import get_grid_search
from polynet.options.enums import ProblemTypes, TransformDescriptors

from polynet.featurizer.preprocess import transform_dependent_variables
from polynet.train.model_utils import generate_models, get_model
from polynet.options.enums import TradtionalMLModels


def train_tml_ensemble(
    tml_models: dict[TradtionalMLModels, dict],
    problem_type: ProblemTypes,
    transform_features: TransformDescriptors,
    dataframes: dict[str, pd.DataFrame],
    general_experiment_options: GeneralConfigOptions,
    data_options: DataOptions,
    train_val_test_idxs: tuple,
):

    train_ids, val_ids, test_ids = deepcopy(train_val_test_idxs)

    # dictionaries to save models, data and scalers for each training process
    trained_models, training_data, scalers = {}, {}, {}

    for i, (train_idxs, val_idxs, test_idxs) in enumerate(zip(train_ids, val_ids, test_ids)):

        iteration = i + 1

        # validation set not required on TML models
        train_idxs += val_idxs

        for df_name, df in dataframes.items():

            train_df = df.loc[train_idxs]
            val_df = df.loc[val_idxs]
            test_df = df.loc[test_idxs]

            log_name = f"{df_name}_{iteration}"

            if transform_features != TransformDescriptors.NoTransformation:

                original_train_features = train_df.iloc[:, :-1].copy()

                train_features, scaler = transform_dependent_variables(
                    fit_data=original_train_features,
                    transform_data=train_df.iloc[:, :-1],
                    transform_type=transform_features,
                )
                train_df.iloc[:, :-1] = train_features

                val_features = scaler.transform(val_df.iloc[:, :-1])
                val_df.iloc[:, :-1] = val_features

                test_features = scaler.transform(test_df.iloc[:, :-1])
                test_df.iloc[:, :-1] = test_features

                scalers[log_name] = scaler

            training_data[log_name] = (train_df, val_df, test_df)

            models = generate_models(models_to_train=tml_models.keys(), problem_type=problem_type)

            for model_name, model in models.items():

                # if there are not hyperparameters set, then assume hpo
                is_hpo = not tml_models[model_name]

                model = get_model(
                    model_type=model,
                    model_params=tml_models[model_name],
                    random_state=general_experiment_options.random_seed + i,
                )

                if is_hpo:
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
