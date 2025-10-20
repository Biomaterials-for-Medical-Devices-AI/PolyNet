from polynet.options.enums import ProblemTypes, TradtionalMLModels, NetworkParams, Pooling, Networks

LINEAR_MODEL_GRID = {"fit_intercept": [True, False]}

LOGISTIC_MODEL_GRID = {
    "penalty": ["l1", "l2"],
    "C": [0.1, 1, 10, 100],
    "fit_intercept": [True, False],
    "solver": ["lbfgs", "liblinear"],
}


RANDOM_FOREST_GRID = {
    "n_estimators": [100, 300, 500],
    "min_samples_split": [2, 0.05, 0.1],
    "min_samples_leaf": [1, 0.05, 0.1],
    "max_depth": [None, 3, 6],
}

XGB_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 3, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.15, 0.20, 0.25],
}

SVM_GRID = {
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "degree": [2, 3, 4],
    "C": [1.0, 10.0, 100],
}

GNN_SHARED_GRID = {
    NetworkParams.PoolingNet: [
        Pooling.GlobalAddPool,
        Pooling.GlobalMaxPool,
        Pooling.GlobalMeanPool,
        Pooling.GlobalMeanMaxPool,
    ],
    NetworkParams.NumConvolutions: [1, 2, 3],
    NetworkParams.EmbeddingDim: [32, 64, 128],
    NetworkParams.ReadoutLayers: [1, 2, 3],
    NetworkParams.Dropout: [0.01, 0.05, 0.1],
    NetworkParams.ApplyWeightingGraph: [True, False],
}

GCN_GRID = {NetworkParams.Improved: [True, False]}
GraphSAGE_GRID = {NetworkParams.Bias: [True, False]}
TransformerGNN_GRID = {NetworkParams.NumHeads: [1, 2, 4]}
GAT_GRID = {NetworkParams.NumHeads: [1, 2, 4]}


def get_grid_search(
    model_name: TradtionalMLModels, problem_type: ProblemTypes.Classification, random_seed: int
):
    """Get the grid search parameters for the specified model."""
    match model_name:

        case TradtionalMLModels.LinearRegression:
            if problem_type == ProblemTypes.Regression:
                return LINEAR_MODEL_GRID
            elif problem_type == ProblemTypes.Classification:
                return LOGISTIC_MODEL_GRID

        case TradtionalMLModels.LogisticRegression:
            LINEAR_MODEL_GRID["random_state"] = [random_seed]
            return LINEAR_MODEL_GRID

        case TradtionalMLModels.RandomForest:
            RANDOM_FOREST_GRID["random_state"] = [random_seed]
            return RANDOM_FOREST_GRID

        case TradtionalMLModels.XGBoost:
            XGB_GRID["random_state"] = [random_seed]
            return XGB_GRID

        case TradtionalMLModels.SupportVectorMachine:
            SVM_GRID["random_state"] = [random_seed]
            if problem_type == ProblemTypes.Classification:
                SVM_GRID["probability"] = [True]
            return SVM_GRID

        case Networks.GCN:
            GCN_GRID["random_state"] = [random_seed]
            return {**GCN_GRID, **GNN_SHARED_GRID}

        case Networks.GraphSAGE:
            GraphSAGE_GRID["random_state"] = [random_seed]
            return {**GraphSAGE_GRID, **GNN_SHARED_GRID}

        case Networks.TransformerGNN:
            TransformerGNN_GRID["random_state"] = [random_seed]
            return {**TransformerGNN_GRID, **GNN_SHARED_GRID}

        case Networks.GAT:
            GAT_GRID["random_state"] = [random_seed]
            return {**GAT_GRID, **GNN_SHARED_GRID}

        case Networks.MPNN:
            MPNN_GRID = {"random_state": [random_seed]}
            return {**MPNN_GRID, **GNN_SHARED_GRID}

        case Networks.CGGNN:
            CGGNN_GRID = {"random_state": [random_seed]}
            return {**CGGNN_GRID, **GNN_SHARED_GRID}

        case _:
            raise ValueError(f"Unknown model type: {model_name}")
