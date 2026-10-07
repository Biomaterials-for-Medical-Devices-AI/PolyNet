"""
polynet.config.search_grid
===========================
Default hyperparameter search grids for hyperparameter optimisation (HPO),
and the merge of user-supplied grids (``hpo_search_grid``) on top of them.

Design notes
------------
* All grid dicts are **templates** — they are never mutated in place.
  The ``get_tml_search_grid`` and ``get_gnn_search_grid`` functions always
  return a *copy* with the random seed injected, so repeated calls with
  different seeds are safe.
* GNN and TML grids are looked up by separate functions to keep the API
  clear and avoid a single overloaded function that accepts both model
  families.
* These grids represent sensible defaults. Users can override individual
  parameters with ``gnn_training.hpo_search_grid`` / ``tml_models.hpo_search_grid``:
  a supplied parameter *replaces* the default candidates, every other parameter
  keeps its defaults (see ``merge_search_grid``).
* Values PolyNet sets itself (``RESERVED_GNN_GRID_KEYS`` /
  ``RESERVED_TML_GRID_KEYS``) cannot be overridden.
"""

import copy
import logging
import math

from polynet.config.enums import (
    ApplyWeightingToGraph,
    ArchitectureParam,
    Network,
    Pooling,
    ProblemType,
    TraditionalMLModel,
    TrainingParam,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Traditional ML grids (templates — never mutate these directly)
# ---------------------------------------------------------------------------

_LINEAR_REGRESSION_GRID: dict = {"fit_intercept": [True, False]}

# sklearn >= 1.8 deprecated ``penalty`` in favour of ``l1_ratio``
# (0.0 = L2, 1.0 = L1, in-between = elastic-net). The ``saga`` solver
# supports the full ``l1_ratio`` range, so there are no invalid
# solver/penalty combinations (avoids FitFailedWarning) and no use of
# the deprecated ``penalty`` arg (avoids FutureWarning). ``max_iter``
# is raised well above the default of 100 to avoid ConvergenceWarning.
_LOGISTIC_REGRESSION_GRID: dict = {
    "solver": ["saga"],
    "l1_ratio": [0.0, 0.5, 1.0],
    "C": [0.1, 1, 10, 100],
    "fit_intercept": [True, False],
    "max_iter": [5000],
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
    ArchitectureParam.ApplyWeightingGraph: [ApplyWeightingToGraph.PerMonomerPooling],
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


# Parameters injected by PolyNet that user grids may not override.
RESERVED_GNN_GRID_KEYS = frozenset({TrainingParam.Seed})

# Default AsymmetricLossStrength candidates searched for classification
# (``None`` = no class weighting). Regression never uses class weights.
CLASSIFICATION_LOSS_STRENGTHS = [None, 0.25, 0.5, 0.75, 1.0]
RESERVED_TML_GRID_KEYS = frozenset({"random_state", "probability"})

# Key of ``gnn_training.hpo_search_grid`` applied to every architecture.
SHARED_GNN_GRID_KEY = "shared"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def merge_search_grid(default: dict, *overrides: dict | None) -> dict:
    """
    Return ``default`` with the parameters of each override replacing its candidates.

    Overrides are applied in order (later ones win). Parameters not present in
    any override keep their default candidates. Neither input is modified.

    Parameters
    ----------
    default:
        Default grid ``{param: [candidates]}``.
    *overrides:
        User grids ``{param: [candidates]}``; ``None`` entries are skipped.

    Returns
    -------
    dict
        The merged grid.
    """
    merged = copy.deepcopy(default)
    for override in overrides:
        for param, values in (override or {}).items():
            merged[param] = list(values)
    return merged


def n_grid_combinations(grid: dict) -> int:
    """Number of distinct configurations in a grid (product of the candidate counts)."""
    return math.prod(len(v) if isinstance(v, list) else 1 for v in grid.values())


def gnn_grid_parameters(network: Network) -> set[str]:
    """Parameters a user grid may set for ``network`` (its default grid minus reserved keys)."""
    return set(get_gnn_search_grid(network, random_seed=0)) - RESERVED_GNN_GRID_KEYS


def shared_gnn_grid_parameters() -> set[str]:
    """Parameters the ``shared`` entry of a user GNN grid may set."""
    return set(_GNN_SHARED_GRID) - RESERVED_GNN_GRID_KEYS


def get_tml_search_grid(
    model: TraditionalMLModel,
    problem_type: ProblemType,
    random_seed: int,
    custom_grid: dict | None = None,
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
    custom_grid:
        Optional user grid for this model (``tml_models.hpo_search_grid[model]``).
        Its parameters replace the default candidates; ``random_state`` /
        ``probability`` are always set by PolyNet.

    Returns
    -------
    dict
        A hyperparameter grid, sampled by sklearn's ``RandomizedSearchCV``.

    Raises
    ------
    ValueError
        If the model is not recognised.
    """
    user = custom_grid or {}
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
            if problem_type == ProblemType.Classification:
                grid["random_state"] = [random_seed]
                grid["probability"] = [True]

        case _:
            raise ValueError(
                f"No TML search grid defined for model '{model}'. "
                f"Available models: {[m.value for m in TraditionalMLModel]}"
            )

    # User candidates replace the defaults; PolyNet-injected values stay authoritative.
    return merge_search_grid(
        grid, {k: v for k, v in user.items() if k not in RESERVED_TML_GRID_KEYS}
    )


def get_gnn_search_grid(
    network: Network,
    random_seed: int,
    problem_type: ProblemType | None = None,
    custom_grid: dict | None = None,
) -> dict:
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
    problem_type:
        For classification, ``AsymmetricLossStrength`` defaults to
        ``CLASSIFICATION_LOSS_STRENGTHS``; for regression it is always
        ``[None]`` (a user value is ignored with a warning).
    custom_grid:
        Optional ``gnn_training.hpo_search_grid``. Its ``shared`` entry and the
        entry for ``network`` replace the default candidates of the parameters
        they set (the architecture entry wins over ``shared``). The seed is
        always set by PolyNet.

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
    if problem_type == ProblemType.Classification:
        shared[TrainingParam.AsymmetricLossStrength] = list(CLASSIFICATION_LOSS_STRENGTHS)
    grid = {**specific, **shared}
    user = custom_grid or {}
    grid = merge_search_grid(
        grid,
        *(
            {k: v for k, v in (user.get(key) or {}).items() if k not in RESERVED_GNN_GRID_KEYS}
            for key in (SHARED_GNN_GRID_KEY, network.value)
        ),
    )
    if problem_type == ProblemType.Regression and grid[TrainingParam.AsymmetricLossStrength] != [None]:
        logger.warning(
            f"hpo_search_grid sets {TrainingParam.AsymmetricLossStrength.value} for "
            f"{network.value}, but class weights only apply to classification; ignoring it."
        )
        grid[TrainingParam.AsymmetricLossStrength] = [None]
    grid[TrainingParam.Seed] = [random_seed]
    return grid


def effective_search_spaces(problem_type: ProblemType, gnn_cfg=None, tml_cfg=None) -> dict:
    """
    Describe the search spaces automatic HPO will use in an experiment.

    For every architecture / model that runs HPO (empty hyperparameter
    block), returns its default grid merged with the user's
    ``hpo_search_grid``, together with the sample count and folds. Seeds
    (``seed`` / ``random_state``) are left out because they change per split
    (``random_seed + split - 1``); everything else is exactly what is searched.

    Parameters
    ----------
    problem_type:
        Classification or regression (TML grids depend on it).
    gnn_cfg:
        ``TrainGNNConfig`` or ``None`` if GNNs are not trained.
    tml_cfg:
        ``TrainTMLConfig`` or ``None`` if TML models are not trained.

    Returns
    -------
    dict
        ``{"gnn": {...}, "tml": {...}}`` with an entry only for pipelines that
        run HPO; empty if nothing is tuned.
    """
    spaces: dict = {}

    if gnn_cfg is not None:
        architectures = {
            net.value: {
                k: v
                for k, v in get_gnn_search_grid(
                    net,
                    random_seed=0,
                    problem_type=problem_type,
                    custom_grid=gnn_cfg.hpo_search_grid,
                ).items()
                if k != TrainingParam.Seed
            }
            for net, params in gnn_cfg.gnn_convolutional_layers.items()
            if not params
        }
        if architectures:
            spaces["gnn"] = {
                "hpo_num_samples": gnn_cfg.hpo_num_samples,
                "hpo_split_strategy": gnn_cfg.hpo_split_strategy.value,
                "hpo_n_folds": gnn_cfg.hpo_n_folds,
                "hpo_val_fraction": gnn_cfg.hpo_val_fraction,
                "hpo_n_repeats": gnn_cfg.hpo_n_repeats,
                "architectures": architectures,
            }

    if tml_cfg is not None:
        models = {}
        for model, params in (tml_cfg.selected_models or {}).items():
            if params:
                continue
            grid = get_tml_search_grid(
                model,
                problem_type,
                random_seed=0,
                custom_grid=tml_cfg.hpo_search_grid.get(model.value),
            )
            models[model.value] = {
                "search_grid": {k: v for k, v in grid.items() if k != "random_state"},
                "n_iter": min(tml_cfg.hpo_num_samples, n_grid_combinations(grid)),
            }
        if models:
            spaces["tml"] = {
                "hpo_num_samples": tml_cfg.hpo_num_samples,
                "hpo_n_folds": tml_cfg.hpo_n_folds,
                "models": models,
            }

    return spaces
