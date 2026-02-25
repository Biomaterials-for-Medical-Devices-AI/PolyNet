import pandas as pd
from polynet.featurizer.graph import PolymerGraphDataset
from torch_geometric.loader import DataLoader

from polynet.config.schemas import DataConfig

from polynet.config.enums import ProblemType
from polynet.config.constants import ResultColumn
from polynet.config.column_names import get_predicted_label_column_name
from polynet.inference.utils import prepare_probs_df
from polynet.models.base import BaseNetwork


def predict_unseen_tml(
    models: dict[str, object],
    scalers: dict[str, object],
    dfs: dict[str, pd.DataFrame],
    data_options: DataConfig,
) -> pd.DataFrame:

    predictions_all = None

    for model_name, model in models.items():

        ml_model, iteration = model_name.rsplit("_", 1)
        ml_algorithm, df_name = ml_model.rsplit("-", 1)

        model_log_name = model_name.replace("_", " ")

        predicted_col_name = get_predicted_label_column_name(
            target_variable_name=data_options.target_variable_name, model_name=model_log_name
        )

        df = dfs[df_name]

        if scalers:
            scaler_name = model_name.rsplit("-", 1)[-1]
            scaler = scalers[scaler_name]
            df_cols = df.columns
            df = scaler.transform(df)
            df = pd.DataFrame(df, columns=df_cols)

        preds = model.predict(df)

        preds_df = pd.DataFrame({predicted_col_name: preds})

        if data_options.problem_type == ProblemType.Classification:
            probs_df = prepare_probs_df(
                probs=model.predict_proba(df),
                target_variable_name=data_options.target_variable_name,
                model_name=model_log_name,
            )
            preds_df[probs_df.columns] = probs_df.to_numpy()

        if predictions_all is None:
            predictions_all = preds_df.copy()
        else:
            predictions_all = pd.concat([predictions_all, preds_df], axis=1)

    return predictions_all


def predict_unseen_gnn(
    models: dict[str, BaseNetwork], dataset: PolymerGraphDataset, data_options: DataConfig
) -> pd.DataFrame:

    predictions_all = None

    for model_name, model in models.items():

        model_name = model_name.replace("_", " ")

        predicted_col_name = get_predicted_label_column_name(
            target_variable_name=data_options.target_variable_name, model_name=model_name
        )

        loader = DataLoader(dataset)

        preds = model.predict_loader(loader)

        preds_df = pd.DataFrame({ResultColumn.INDEX: preds[0], predicted_col_name: preds[1]})

        if data_options.problem_type == ProblemType.Classification:
            probs_df = prepare_probs_df(
                probs=preds[-1],
                target_variable_name=data_options.target_variable_name,
                model_name=model_name,
            )
            preds_df[probs_df.columns] = probs_df.to_numpy()

        if predictions_all is None:
            predictions_all = preds_df.copy()
        else:
            predictions_all = pd.merge(
                left=predictions_all, right=preds_df, on=[ResultColumn.INDEX]
            )

    return predictions_all
