"""
polynet.config.search_grids
============================
Default hyperparameter search grids for grid-search optimisation.

Design notes
------------
* All grid dicts are **templates** — they are never mutated in place.
  The ``get_tml_search_grid`` and ``get_gnn_search_grid`` functions always
  return a *copy* with the random seed injected, so repeated calls with
  different seeds are safe.
* GNN and TML grids are looked up by separate functions to keep the API
  clear and avoid a single overloaded function that accepts both model
  families.
* These grids represent sensible defaults. Users can override them by
  supplying a custom grid to the trainer directly.
"""

import copy

from polynet.config.enums import (
    ApplyWeightingToGraph,
    ArchitectureParam,
    Network,
    Pooling,
    ProblemType,
    TrainingParam,
    TraditionalMLModel,
)


# ---------------------------------------------------------------------------
# Traditional ML grids (templates — never mutate these directly)
# ---------------------------------------------------------------------------

_LINEAR_REGRESSION_GRID: dict = {"fit_intercept": [True, False]}

_LOGISTIC_REGRESSION_GRID: dict = {
    "penalty": ["l1", "l2"],
    "C": [0.1, 1, 10, 100],
    "fit_intercept": [True, False],
    "solver": ["lbfgs", "liblinear"],
}

_RANDOM_FOREST_GRID: dict = {
    "n_estimators": [100, 300, 500],
    "min_samples_split": [2, 0.05, 0.1],
    "min_samples_leaf": [1, 0.05, 0.1],
    "max_depth": [None, 3, 6],
}

_XGB_GRID: dict = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 3, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.15, 0.20, 0.25],
}

_SVM_GRID: dict = {
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "degree": [2, 3, 4],
    "C": [1.0, 10.0, 100],
}


# ---------------------------------------------------------------------------
# GNN grids (templates — never mutate these directly)
# ---------------------------------------------------------------------------

_GNN_SHARED_GRID: dict = {
    ArchitectureParam.PoolingMethod: [
        Pooling.GlobalAddPool,
        Pooling.GlobalMaxPool,
        Pooling.GlobalMeanPool,
        Pooling.GlobalMeanMaxPool,
    ],
    ArchitectureParam.NumConvolutions: [1, 2, 3],
    ArchitectureParam.EmbeddingDim: [32, 64, 128],
    ArchitectureParam.ReadoutLayers: [1, 2, 3],
    ArchitectureParam.Dropout: [0.01, 0.05, 0.1],
    ArchitectureParam.ApplyWeightingGraph: [ApplyWeightingToGraph.BeforePooling],
    TrainingParam.LearningRate: [0.0001, 0.001, 0.01],
    TrainingParam.BatchSize: [16, 32, 64],
    TrainingParam.AsymmetricLossStrength: [None],
}

_GCN_GRID: dict = {ArchitectureParam.Improved: [True, False]}
_GraphSAGE_GRID: dict = {ArchitectureParam.Bias: [True, False]}
_TransformerGNN_GRID: dict = {ArchitectureParam.NumHeads: [1, 2, 4]}
_GAT_GRID: dict = {ArchitectureParam.NumHeads: [1, 2, 4]}
_MPNN_GRID: dict = {}
_CGGNN_GRID: dict = {}

_GNN_SPECIFIC_GRIDS: dict[Network, dict] = {
    Network.GCN: _GCN_GRID,
    Network.GraphSAGE: _GraphSAGE_GRID,
    Network.TransformerGNN: _TransformerGNN_GRID,
    Network.GAT: _GAT_GRID,
    Network.MPNN: _MPNN_GRID,
    Network.CGGNN: _CGGNN_GRID,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_tml_search_grid(
    model: TraditionalMLModel, problem_type: ProblemType, random_seed: int
) -> dict:
    """
    Return a hyperparameter search grid for a traditional ML model.

    Always returns a fresh copy — safe to call multiple times with
    different seeds or problem types without side effects.

    Parameters
    ----------
    model:
        The traditional ML model to retrieve a grid for.
    problem_type:
        The task type. Affects which grid is returned for models that
        support both regression and classification (e.g. LinearRegression).
    random_seed:
        Injected into the grid as ``random_state`` where applicable.

    Returns
    -------
    dict
        A hyperparameter grid suitable for use with sklearn's GridSearchCV.

    Raises
    ------
    ValueError
        If the model is not recognised.
    """
    match model:
        case TraditionalMLModel.LinearRegression:
            grid = copy.deepcopy(
                _LOGISTIC_REGRESSION_GRID
                if problem_type == ProblemType.Classification
                else _LINEAR_REGRESSION_GRID
            )

        case TraditionalMLModel.LogisticRegression:
            grid = copy.deepcopy(_LOGISTIC_REGRESSION_GRID)
            grid["random_state"] = [random_seed]

        case TraditionalMLModel.RandomForest:
            grid = copy.deepcopy(_RANDOM_FOREST_GRID)
            grid["random_state"] = [random_seed]

        case TraditionalMLModel.XGBoost:
            grid = copy.deepcopy(_XGB_GRID)
            grid["random_state"] = [random_seed]

        case TraditionalMLModel.SupportVectorMachine:
            grid = copy.deepcopy(_SVM_GRID)
            grid["random_state"] = [random_seed]
            if problem_type == ProblemType.Classification:
                grid["probability"] = [True]

        case _:
            raise ValueError(
                f"No TML search grid defined for model '{model}'. "
                f"Available models: {[m.value for m in TraditionalMLModel]}"
            )

    return grid


def get_gnn_search_grid(network: Network, random_seed: int) -> dict:
    """
    Return a hyperparameter search grid for a GNN architecture.

    Merges the architecture-specific grid with the shared GNN grid and
    injects the random seed. Always returns a fresh copy.

    Parameters
    ----------
    network:
        The GNN architecture to retrieve a grid for.
    random_seed:
        Injected into the grid as ``TrainingParam.Seed``.

    Returns
    -------
    dict
        A combined hyperparameter grid for the specified GNN.

    Raises
    ------
    ValueError
        If the network is not recognised.
    """
    if network not in _GNN_SPECIFIC_GRIDS:
        raise ValueError(
            f"No GNN search grid defined for network '{network}'. "
            f"Available networks: {[n.value for n in Network]}"
        )

    specific = copy.deepcopy(_GNN_SPECIFIC_GRIDS[network])
    shared = copy.deepcopy(_GNN_SHARED_GRID)
    grid = {**specific, **shared, TrainingParam.Seed: [random_seed]}
    return grid
