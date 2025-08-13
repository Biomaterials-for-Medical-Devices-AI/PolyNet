from polynet.options.enums import ProblemTypes, TradtionalMLModels

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


def get_grid_search(
    model_name: TradtionalMLModels, problem_type: ProblemTypes.Classification, random_seed: int
):
    """Get the grid search parameters for the specified model."""
    if model_name == TradtionalMLModels.LinearRegression:
        if problem_type == ProblemTypes.Regression:
            return LINEAR_MODEL_GRID
        elif problem_type == ProblemTypes.Classification:
            return LOGISTIC_MODEL_GRID

    if model_name == TradtionalMLModels.LogisticRegression:
        LINEAR_MODEL_GRID["random_state"] = [random_seed]
        return LINEAR_MODEL_GRID
    elif model_name == TradtionalMLModels.RandomForest:
        RANDOM_FOREST_GRID["random_state"] = [random_seed]
        return RANDOM_FOREST_GRID
    elif model_name == TradtionalMLModels.XGBoost:
        XGB_GRID["random_state"] = [random_seed]
        return XGB_GRID
    elif model_name == TradtionalMLModels.SupportVectorMachine:
        SVM_GRID["random_state"] = [random_seed]
        if problem_type == ProblemTypes.Classification:
            SVM_GRID["probability"] = [True]
        return SVM_GRID
    else:
        raise ValueError(f"Unknown model type: {model_name}")
