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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from polynet.config.enums import (
    FeatureSelection,
    ProblemType,
    TargetTransformDescriptor,
    TraditionalMLModel,
    TransformDescriptor,
)
from polynet.config.search_grid import get_tml_search_grid, n_grid_combinations
from polynet.data.feature_transformer import FeatureTransformer
from polynet.data.preprocessing import TargetScaler
from polynet.training.cv import make_kfold

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
    hpo_num_samples: int = 30,
    hpo_search_grid: dict | None = None,
    include_validation_in_training: bool = True,
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
    hpo_n_folds:
        Cross-validation folds used to score HPO configurations.
    hpo_num_samples:
        Configurations sampled per randomised search (``n_iter``); capped at
        the number of distinct grid combinations.
    hpo_search_grid:
        User search-grid candidates keyed by model name
        (``tml_models.hpo_search_grid``), merged on top of the default grids.
    include_validation_in_training:
        If True, the validation samples are added to the training samples
        (feature transformer, target scaler, hyperparameter search and model
        are fitted on both). If False, everything is fitted on the training
        samples only and the validation samples are kept as a held-out set.

    Returns
    -------
    tuple[dict, dict, dict, dict]
        ``(trained_models, training_data, scalers, target_scalers)`` where:
        - ``trained_models``: ``{model_log_name: fitted_model}``
        - ``training_data``: ``{log_name: (train_df, val_df, test_df)}``;
          ``val_df`` is ``None`` when the validation samples were used for
          training
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

    logger.info(
        "TML models are trained on the "
        + ("training + validation samples." if include_validation_in_training else "training samples only (validation held out, as for GNNs).")
    )
    train_ids, val_ids, test_ids = deepcopy(train_val_test_idxs)
    trained_models: dict = {}
    training_data: dict = {}
    scalers: dict = {}
    target_scalers: dict = {}

    for i, (train_idxs, val_idxs, test_idxs) in enumerate(zip(train_ids, val_ids, test_ids)):
        iteration = i + 1
        seed = random_seed + i

        # Either merge the validation samples into training, or keep them as a
        # held-out set (training samples only, like the GNNs).
        val_idxs = pd.Index(val_idxs) if val_idxs is not None else pd.Index([])
        if include_validation_in_training:
            combined_train_idxs = pd.Index(train_idxs).append(val_idxs)
            held_out_val_idxs = None
        else:
            combined_train_idxs = pd.Index(train_idxs)
            held_out_val_idxs = val_idxs

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

            val_df = None
            if held_out_val_idxs is not None and len(held_out_val_idxs):
                val_raw = df.loc[held_out_val_idxs]
                X_val = pd.DataFrame(
                    transformer.transform(val_raw.iloc[:, :-1]),
                    index=held_out_val_idxs,
                    columns=transformer.get_feature_names_out(),
                )
                val_df = pd.concat([X_val, val_raw.iloc[:, -1]], axis=1)

            scalers[log_name] = transformer
            # training_data always stores original (unscaled) y so that y_true
            # in the predictions DataFrame is always in the original target range.
            training_data[log_name] = (train_df, val_df, test_df)

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
                        n_iter=hpo_num_samples,
                        custom_grid=(hpo_search_grid or {}).get(model_id.value),
                    )
                else:
                    logger.info(f"Fitting {model_id.value} (iteration {iteration}, {df_name})...")
                    model.fit(train_df_fit.iloc[:, :-1], train_df_fit.iloc[:, -1])

                model_log_name = f"{model_id.value.replace(' ', '')}-{log_name}"
                trained_models[model_log_name] = model

    return trained_models, training_data, scalers, target_scalers


def tml_search_space(
    model_id: TraditionalMLModel,
    problem_type: ProblemType,
    random_seed: int,
    n_iter: int,
    custom_grid: dict | None = None,
) -> tuple[dict, int]:
    """
    Return the grid and number of samples for a TML randomised search.

    The grid is the default grid of ``model_id`` with the user candidates
    (``custom_grid``) merged on top. ``n_iter`` is capped at the number of
    distinct grid combinations — sampling more would only repeat
    configurations — with a warning.

    Returns
    -------
    tuple[dict, int]
        ``(grid, n_iter_used)``.
    """
    grid = get_tml_search_grid(
        model=model_id, problem_type=problem_type, random_seed=random_seed, custom_grid=custom_grid
    )
    n_combinations = n_grid_combinations(grid)
    if n_iter > n_combinations:
        logger.warning(
            f"hpo_num_samples={n_iter} exceeds the {n_combinations} distinct configurations of "
            f"the {model_id.value} search grid; sampling all {n_combinations} instead."
        )
        n_iter = n_combinations
    return grid, n_iter


def _run_random_search(
    model,
    model_id: TraditionalMLModel,
    train_df: pd.DataFrame,
    problem_type: ProblemType,
    random_seed: int,
    n_folds: int = 5,
    n_iter: int = 30,
    custom_grid: dict | None = None,
) -> object:
    """
    Tune a TML model with a randomised hyperparameter search.

    Samples ``n_iter`` configurations from the model's search grid (defaults
    merged with ``custom_grid``, see ``tml_search_space``) with
    ``RandomizedSearchCV`` and scores them by ``n_folds``-fold shuffled
    cross-validation (stratified for classification). A summary of the search
    is attached to the returned estimator as ``polynet_hpo_`` (search grid,
    samples used, folds, best parameters and CV score) for provenance.

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
        Number of CV folds. Checked against the data before training starts
        (``polynet.utils.validation.validate_hpo_folds``).
    n_iter:
        Number of configurations to sample (``tml_models.hpo_num_samples``);
        capped at the number of distinct grid combinations.
    custom_grid:
        User candidates for this model (``tml_models.hpo_search_grid[model]``).

    Returns
    -------
    object
        The best estimator, refitted on the full ``train_df``.
    """
    param_grid, n_iter = tml_search_space(
        model_id=model_id,
        problem_type=problem_type,
        random_seed=random_seed,
        n_iter=n_iter,
        custom_grid=custom_grid,
    )
    cv = make_kfold(problem_type=problem_type, n_folds=n_folds, random_seed=random_seed)

    logger.info(
        f"Running RandomizedSearchCV for {model_id.value} "
        f"({n_iter} samples, {n_folds}-fold CV, seed={random_seed})..."
    )

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        random_state=random_seed,
        n_jobs=-1,
    )
    random_search.fit(train_df.iloc[:, :-1], train_df.iloc[:, -1])

    logger.info(f"Best params for {model_id.value}: {random_search.best_params_}")
    best = random_search.best_estimator_
    best.polynet_hpo_ = {
        "search_grid": param_grid,
        "n_iter": n_iter,
        "n_folds": n_folds,
        "seed": random_seed,
        "best_params": random_search.best_params_,
        "best_cv_score": float(random_search.best_score_),
    }
    return best
