from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from polynet.options.enums import ProblemTypes, TradtionalMLModels


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


def generate_models(models_to_train: list, problem_type: ProblemTypes):

    models = {}

    if TradtionalMLModels.LinearRegression in models_to_train:
        models[TradtionalMLModels.LinearRegression] = LinearRegression

    if TradtionalMLModels.LogisticRegression in models_to_train:
        models[TradtionalMLModels.LogisticRegression] = LogisticRegression

    if TradtionalMLModels.RandomForest in models_to_train:
        if problem_type == ProblemTypes.Classification:
            models[TradtionalMLModels.RandomForest] = RandomForestClassifier
        elif problem_type == ProblemTypes.Regression:
            models[TradtionalMLModels.RandomForest] = RandomForestRegressor

    if TradtionalMLModels.SupportVectorMachine in models_to_train:
        if problem_type == ProblemTypes.Classification:
            models[TradtionalMLModels.SupportVectorMachine] = SVC
        elif problem_type == ProblemTypes.Regression:
            models[TradtionalMLModels.SupportVectorMachine] = SVR

    if TradtionalMLModels.XGBoost in models_to_train:
        if problem_type == ProblemTypes.Classification:
            models[TradtionalMLModels.XGBoost] = XGBClassifier
        elif problem_type == ProblemTypes.Regression:
            models[TradtionalMLModels.XGBoost] = XGBRegressor

    return models
