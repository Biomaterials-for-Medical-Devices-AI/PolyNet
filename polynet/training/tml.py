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

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from polynet.config.enums import (
    FeatureSelection,
    ProblemType,
    TargetTransformDescriptor,
    TraditionalMLModel,
    TransformDescriptor,
)
from polynet.config.search_grid import get_tml_search_grid
from polynet.data.feature_transformer import FeatureTransformer
from polynet.data.preprocessing import TargetScaler

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
    feature_selection: dict[FeatureSelection, dict],
    dataframes: dict[str, pd.DataFrame],
    random_seed: int,
    train_val_test_idxs: tuple[list, list | None, list],
    target_transform: TargetTransformDescriptor | str = TargetTransformDescriptor.NoTransformation,
    hpo_n_folds: int = 5,
) -> tuple[dict, dict, dict, dict]:
    """
    Train an ensemble of TML models across all bootstrap iterations.

    For each iteration and each descriptor DataFrame, each requested model
    is either fitted with the provided hyperparameters or tuned via a
    randomised search (``RandomizedSearchCV``, ``hpo_n_folds``-fold shuffled
    CV) if no hyperparameters are provided.

    Note: The validation set is merged into the training set for TML
    models, as TML training uses internal cross-validation for HPO
    rather than a held-out validation set.

    Parameters
    ----------
    tml_models:
        Mapping from model identifier to hyperparameter dict. Pass an
        empty dict or ``None`` to trigger randomised-search HPO for that model.
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
    target_transform:
        Optional scaling strategy for the target variable. The scaler is
        fitted on the training set only; ``training_data`` always stores
        the original (unscaled) target values so that ``y_true`` in the
        predictions DataFrame is always in the original range.

    Returns
    -------
    tuple[dict, dict, dict, dict]
        ``(trained_models, training_data, scalers, target_scalers)`` where:
        - ``trained_models``: ``{model_log_name: fitted_model}``
        - ``training_data``: ``{log_name: (train_df, test_df)}``
        - ``scalers``: ``{log_name: fitted_feature_scaler}`` or empty dict
        - ``target_scalers``: ``{log_name: TargetScaler}``
    """
    problem_type = ProblemType(problem_type) if isinstance(problem_type, str) else problem_type
    transform_type = (
        TransformDescriptor(transform_type) if isinstance(transform_type, str) else transform_type
    )
    target_transform = (
        TargetTransformDescriptor(target_transform)
        if isinstance(target_transform, str)
        else target_transform
    )

    train_ids, val_ids, test_ids = deepcopy(train_val_test_idxs)
    trained_models: dict = {}
    training_data: dict = {}
    scalers: dict = {}
    target_scalers: dict = {}

    # Fail fast: check the HPO fold count against every split before fitting anything.
    if any(not params for params in tml_models.values()):
        any_df = next(iter(dataframes.values()))
        validate_hpo_n_folds_for_splits(
            n_folds=hpo_n_folds,
            y=any_df.iloc[:, -1],
            train_val_test_idxs=(train_ids, val_ids, test_ids),
            problem_type=problem_type,
        )

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

            X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
            X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]

            logger.info(
                "[%s] iteration %d: fitting feature transformer on %d training samples, "
                "%d features.",
                df_name,
                iteration,
                len(X_train),
                X_train.shape[1],
            )
            transformer = FeatureTransformer(scaler=transform_type, selectors=feature_selection)
            transformer.fit(X_train)

            X_train = transformer.transform(X_train)
            X_train = pd.DataFrame(
                X_train, index=combined_train_idxs, columns=transformer.get_feature_names_out()
            )
            train_df = pd.concat([X_train, y_train], axis=1)

            X_test = transformer.transform(X_test)
            X_test = pd.DataFrame(
                X_test, index=test_idxs, columns=transformer.get_feature_names_out()
            )
            test_df = pd.concat([X_test, y_test], axis=1)

            scalers[log_name] = transformer
            # training_data always stores original (unscaled) y so that y_true
            # in the predictions DataFrame is always in the original target range.
            training_data[log_name] = (train_df, test_df)

            # Fit target scaler on training y; models are trained on scaled y.
            target_scaler = TargetScaler(strategy=target_transform)
            if (
                problem_type == ProblemType.Regression
                and target_transform != TargetTransformDescriptor.NoTransformation
            ):
                target_scaler.fit(y_train.values)
            target_scalers[log_name] = target_scaler

            y_train_fit = (
                np.array(target_scaler.transform(y_train.values))
                if problem_type == ProblemType.Regression
                and target_transform != TargetTransformDescriptor.NoTransformation
                else y_train.values
            )

            model_classes = generate_models(
                models_to_train=list(tml_models.keys()), problem_type=problem_type
            )

            # Build a training DataFrame with scaled y for model fitting, while
            # keeping training_data (returned to the caller) in original scale.
            train_df_fit = pd.concat(
                [X_train, pd.Series(y_train_fit, index=combined_train_idxs, name=y_train.name)],
                axis=1,
            )

            for model_id, model_cls in model_classes.items():
                model_params = tml_models[model_id]
                is_hpo = not model_params

                model = get_model(model_cls=model_cls, model_params=model_params, random_state=seed)

                if is_hpo:
                    model = _run_random_search(
                        model=model,
                        model_id=model_id,
                        train_df=train_df_fit,
                        problem_type=problem_type,
                        random_seed=seed,
                        n_folds=hpo_n_folds,
                    )
                else:
                    logger.info(f"Fitting {model_id.value} (iteration {iteration}, {df_name})...")
                    model.fit(train_df_fit.iloc[:, :-1], train_df_fit.iloc[:, -1])

                model_log_name = f"{model_id.value.replace(' ', '')}-{log_name}"
                trained_models[model_log_name] = model

    return trained_models, training_data, scalers, target_scalers


def validate_hpo_n_folds(n_folds: int, y, problem_type: ProblemType) -> None:
    """
    Check that ``n_folds`` is usable for cross-validation on targets ``y``.

    Parameters
    ----------
    n_folds:
        Requested number of CV folds (``k``).
    y:
        Training targets the search will be run on.
    problem_type:
        Classification or regression.

    Raises
    ------
    ValueError
        If ``k < 2``, if ``k`` exceeds the number of training samples, or,
        for classification, if any class has fewer than ``k`` training
        samples (stratified folds need every class in every fold).
    """
    y = pd.Series(np.asarray(y).ravel())
    n_samples = len(y)

    if n_folds < 2:
        raise ValueError(f"hpo_n_folds must be at least 2, got {n_folds}.")
    if n_folds > n_samples:
        raise ValueError(
            f"hpo_n_folds={n_folds} exceeds the number of training samples ({n_samples}). "
            f"Choose hpo_n_folds between 2 and {n_samples}."
        )
    if problem_type == ProblemType.Classification:
        class_counts = y.value_counts()
        smallest_class, smallest_count = class_counts.idxmin(), int(class_counts.min())
        if n_folds > smallest_count:
            raise ValueError(
                f"hpo_n_folds={n_folds} is larger than the smallest class in the training set "
                f"(class {smallest_class} has {smallest_count} samples). Stratified CV needs "
                f"every class in every fold: choose hpo_n_folds between 2 and {smallest_count}."
            )


def validate_hpo_n_folds_for_splits(
    n_folds: int, y: pd.Series, train_val_test_idxs: tuple, problem_type: ProblemType
) -> None:
    """
    Run ``validate_hpo_n_folds`` on the HPO data of every split.

    TML hyperparameter search runs on the training + validation samples of
    each split, so ``k`` is checked against exactly those samples.

    Parameters
    ----------
    n_folds:
        Requested number of CV folds (``k``).
    y:
        Target values for the whole dataset, indexed by sample ID.
    train_val_test_idxs:
        ``(train_ids, val_ids, test_ids)``, each a list with one entry per split.
    problem_type:
        Classification or regression.

    Raises
    ------
    ValueError
        If ``k`` is invalid for any split; the message names the split.
    """
    train_ids, val_ids, _ = train_val_test_idxs
    for i, (train_idxs, val_idxs) in enumerate(zip(train_ids, val_ids), start=1):
        hpo_idxs = pd.Index(train_idxs).append(pd.Index(val_idxs if val_idxs is not None else []))
        try:
            validate_hpo_n_folds(n_folds=n_folds, y=y.loc[hpo_idxs], problem_type=problem_type)
        except ValueError as e:
            raise ValueError(f"tml_models.hpo_n_folds is invalid for split {i}: {e}") from None


def _make_hpo_cv(problem_type: ProblemType, random_seed: int, n_splits: int = 5):
    """
    Return the shuffled K-fold splitter used for TML hyperparameter search.

    Folds are always shuffled so that a dataset sorted by target (or by any
    other column) does not produce biased folds. Classification uses
    stratified folds to preserve class proportions.
    """
    if problem_type == ProblemType.Classification:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)


def _run_random_search(
    model,
    model_id: TraditionalMLModel,
    train_df: pd.DataFrame,
    problem_type: ProblemType,
    random_seed: int,
    n_folds: int = 5,
) -> object:
    """
    Tune a TML model with a randomised hyperparameter search.

    Samples 30 configurations from the model's search grid
    (``get_tml_search_grid``) with ``RandomizedSearchCV`` and scores them by
    ``n_folds``-fold shuffled cross-validation (stratified for classification).

    Parameters
    ----------
    model:
        Unfitted estimator to tune.
    model_id:
        Model identifier, used to look up the search grid.
    train_df:
        Training data: features in all columns except the last, target last.
    problem_type:
        Classification or regression.
    random_seed:
        Seed for the fold shuffling and the configuration sampling.
    n_folds:
        Number of CV folds; validated against ``train_df`` (see
        ``validate_hpo_n_folds``).

    Returns
    -------
    object
        The best estimator, refitted on the full ``train_df``.
    """
    param_grid = get_tml_search_grid(
        model=model_id, problem_type=problem_type, random_seed=random_seed
    )
    validate_hpo_n_folds(n_folds=n_folds, y=train_df.iloc[:, -1], problem_type=problem_type)
    cv = _make_hpo_cv(problem_type=problem_type, random_seed=random_seed, n_splits=n_folds)

    logger.info(
        f"Running RandomizedSearchCV for {model_id.value} "
        f"({n_folds}-fold CV, seed={random_seed})..."
    )

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=30,
        cv=cv,
        random_state=random_seed,
        n_jobs=-1,
    )
    random_search.fit(train_df.iloc[:, :-1], train_df.iloc[:, -1])

    logger.info(f"Best params for {model_id.value}: {random_search.best_params_}")
    return random_search.best_estimator_
