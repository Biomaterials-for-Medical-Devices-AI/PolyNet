"""
polynet.training.tml
=====================
Traditional machine learning model instantiation and ensemble training.

Supports Random Forest, XGBoost, SVM, Logistic Regression, and Linear
Regression for both classification and regression tasks.

Public API
----------
::

    from polynet.training.tml import generate_models, train_tml_ensemble
"""

from __future__ import annotations

from copy import deepcopy
import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from polynet.config.enums import ProblemType, TraditionalMLModel, TransformDescriptor
from polynet.config.search_grid import get_tml_search_grid
from polynet.data.preprocessing import transform_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Maps (TraditionalMLModel, ProblemType) → sklearn class
# Adding a new model requires one entry per task type.
_TML_REGISTRY: dict[tuple[TraditionalMLModel, ProblemType], type] = {
    (TraditionalMLModel.LinearRegression, ProblemType.Regression): LinearRegression,
    (TraditionalMLModel.LogisticRegression, ProblemType.Classification): LogisticRegression,
    (TraditionalMLModel.RandomForest, ProblemType.Classification): RandomForestClassifier,
    (TraditionalMLModel.RandomForest, ProblemType.Regression): RandomForestRegressor,
    (TraditionalMLModel.SupportVectorMachine, ProblemType.Classification): SVC,
    (TraditionalMLModel.SupportVectorMachine, ProblemType.Regression): SVR,
    (TraditionalMLModel.XGBoost, ProblemType.Classification): XGBClassifier,
    (TraditionalMLModel.XGBoost, ProblemType.Regression): XGBRegressor,
}

# Models that do not accept a random_state constructor argument
_NO_RANDOM_STATE = {LinearRegression, SVR, SVC}


def generate_models(
    models_to_train: list[TraditionalMLModel], problem_type: ProblemType | str
) -> dict[TraditionalMLModel, type]:
    """
    Return a mapping from model identifier to its sklearn class.

    Parameters
    ----------
    models_to_train:
        List of ``TraditionalMLModel`` enum members to include.
    problem_type:
        Determines which variant (classifier vs regressor) is selected.

    Returns
    -------
    dict[TraditionalMLModel, type]
        Mapping from model identifier to uninstantiated sklearn class.

    Raises
    ------
    ValueError
        If a requested model does not support the given problem type.
    """
    problem_type = ProblemType(problem_type) if isinstance(problem_type, str) else problem_type
    result: dict[TraditionalMLModel, type] = {}

    for model in models_to_train:
        key = (model, problem_type)
        if key not in _TML_REGISTRY:
            raise ValueError(
                f"Model '{model.value}' does not support problem type '{problem_type.value}'. "
                f"Available combinations: {[(m.value, p.value) for m, p in _TML_REGISTRY]}."
            )
        result[model] = _TML_REGISTRY[key]

    return result


def get_model(model_cls: type, model_params: dict | None, random_state: int) -> object:
    """
    Instantiate a TML model with the given parameters.

    Parameters
    ----------
    model_cls:
        The sklearn model class to instantiate.
    model_params:
        Hyperparameter dict passed to the constructor. If ``None``,
        the model is instantiated with defaults (for HPO workflows).
    random_state:
        Random seed. Injected automatically unless the model class
        does not accept ``random_state`` (e.g. ``LinearRegression``).

    Returns
    -------
    object
        An instantiated sklearn-compatible model.
    """
    if model_params is None:
        return model_cls()

    params = dict(model_params)
    if model_cls not in _NO_RANDOM_STATE:
        params["random_state"] = random_state

    return model_cls(**params)


# ---------------------------------------------------------------------------
# Ensemble training
# ---------------------------------------------------------------------------


def train_tml_ensemble(
    tml_models: dict[TraditionalMLModel, dict | None],
    problem_type: ProblemType | str,
    transform_type: TransformDescriptor | str,
    dataframes: dict[str, pd.DataFrame],
    random_seed: int,
    train_val_test_idxs: tuple[list, list | None, list],
) -> tuple[dict, dict, dict]:
    """
    Train an ensemble of TML models across all bootstrap iterations.

    For each iteration and each descriptor DataFrame, each requested model
    is either fitted with the provided hyperparameters or tuned via
    ``GridSearchCV`` if no hyperparameters are provided.

    Note: The validation set is merged into the training set for TML
    models, as TML training uses internal cross-validation for HPO
    rather than a held-out validation set.

    Parameters
    ----------
    tml_models:
        Mapping from model identifier to hyperparameter dict. Pass an
        empty dict or ``None`` to trigger GridSearchCV HPO for that model.
    problem_type:
        Classification or regression.
    transform_type:
        Feature scaling to apply before training. Use
        ``TransformDescriptor.NoTransformation`` to skip scaling.
    dataframes:
        Dict of ``{descriptor_set_name: DataFrame}`` where each DataFrame
        has features in all columns except the last, and the target in
        the last column.
    random_seed:
        Base random seed. Each iteration uses ``random_seed + i``.
    train_val_test_idxs:
        Triple of ``(train_indices, val_indices, test_indices)`` as
        returned by ``get_data_split_indices``.

    Returns
    -------
    tuple[dict, dict, dict]
        ``(trained_models, training_data, scalers)`` where:
        - ``trained_models``: ``{model_log_name: fitted_model}``
        - ``training_data``: ``{log_name: (train_df, test_df)}``
        - ``scalers``: ``{log_name: fitted_scaler}`` or empty dict
    """
    problem_type = ProblemType(problem_type) if isinstance(problem_type, str) else problem_type
    transform_type = (
        TransformDescriptor(transform_type) if isinstance(transform_type, str) else transform_type
    )

    train_ids, val_ids, test_ids = deepcopy(train_val_test_idxs)
    trained_models: dict = {}
    training_data: dict = {}
    scalers: dict = {}

    for i, (train_idxs, val_idxs, test_idxs) in enumerate(zip(train_ids, val_ids, test_ids)):
        iteration = i + 1
        seed = random_seed + i

        # TML does not use a separate validation set — merge val into train
        val_idxs = pd.Index(val_idxs) if val_idxs is not None else pd.Index([])
        combined_train_idxs = train_idxs.append(val_idxs)

        for df_name, df in dataframes.items():
            log_name = f"{df_name}_{iteration}"

            train_df = df.loc[combined_train_idxs].copy()
            test_df = df.loc[test_idxs].copy()

            if transform_type != TransformDescriptor.NoTransformation:

                fit_features = train_df.iloc[:, :-1].copy()
                X_train_scaled, scaler = transform_features(
                    fit_data=fit_features,
                    transform_data=train_df.iloc[:, :-1],
                    transform_type=transform_type,
                )
                train_df.iloc[:, :-1] = X_train_scaled

                X_test_scaled = scaler.transform(test_df.iloc[:, :-1])
                test_df.iloc[:, :-1] = X_test_scaled
                scalers[log_name] = scaler

            training_data[log_name] = (train_df, test_df)

            model_classes = generate_models(
                models_to_train=list(tml_models.keys()), problem_type=problem_type
            )

            for model_id, model_cls in model_classes.items():
                model_params = tml_models[model_id]
                is_hpo = not model_params

                model = get_model(model_cls=model_cls, model_params=model_params, random_state=seed)

                if is_hpo:
                    model = _run_grid_search(
                        model=model,
                        model_id=model_id,
                        train_df=train_df,
                        problem_type=problem_type,
                        random_seed=seed,
                    )
                else:
                    logger.info(f"Fitting {model_id.value} (iteration {iteration}, {df_name})...")
                    model.fit(train_df.iloc[:, :-1], train_df.iloc[:, -1])

                model_log_name = f"{model_id.value.replace(' ', '')}-{log_name}"
                trained_models[model_log_name] = model

    return trained_models, training_data, scalers


def _run_grid_search(
    model,
    model_id: TraditionalMLModel,
    train_df: pd.DataFrame,
    problem_type: ProblemType,
    random_seed: int,
) -> object:
    """Run GridSearchCV HPO and return the best estimator."""
    param_grid = get_tml_search_grid(
        model=model_id, problem_type=problem_type, random_seed=random_seed
    )

    cv = (
        StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
        if problem_type == ProblemType.Classification
        else 5
    )

    logger.info(f"Running GridSearchCV for {model_id.value} (seed={random_seed})...")

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1)
    grid_search.fit(train_df.iloc[:, :-1], train_df.iloc[:, -1])

    logger.info(f"Best params for {model_id.value}: {grid_search.best_params_}")
    return grid_search.best_estimator_
