from polynet.options.enums import TradtionalMLModels


LINEAR_MODEL_GRID = {"fit_intercept": [True, False]}

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


models_grid_search = {
    TradtionalMLModels.LinearRegression: LINEAR_MODEL_GRID,
    TradtionalMLModels.LogisticRegression: LINEAR_MODEL_GRID,
    TradtionalMLModels.RandomForest: RANDOM_FOREST_GRID,
    TradtionalMLModels.XGBoost: XGB_GRID,
    TradtionalMLModels.SupportVectorMachine: SVM_GRID,
}
