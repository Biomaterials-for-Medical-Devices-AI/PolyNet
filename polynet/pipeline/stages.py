"""
polynet.pipeline.stages
========================
Shared pipeline stage functions used by both ``scripts/run_pipeline.py``
and the Streamlit app (``polynet/app/``).

Every function accepts **Pydantic config objects** — never raw dicts — so
validation and type-safety are guaranteed regardless of the entry point.
Stage functions are Streamlit-free; logging uses the standard library so
output is visible in the script and silent (no handler configured) in the app.

Design invariants
-----------------
- ``compute_descriptors`` always saves CSVs **with the pandas index** and
  returns **sanitised** DataFrames. This eliminates the ``index=False`` bug
  that caused the app and the script to produce different training data.
- ``build_graph_dataset`` always saves the GNN raw CSV with ``index=False``
  and a leading ``id_col`` column, so ``pd.read_csv(..., index_col=0)`` in
  the training page recovers the sample ID as the DataFrame index.
- Model files are written inside each ``train_*`` function so callers never
  need to repeat the save loop.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from polynet.config.schemas import (
    DataConfig,
    FeatureTransformConfig,
    RepresentationConfig,
    SplitConfig,
    TrainGNNConfig,
    TrainTMLConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Representation stages
# ---------------------------------------------------------------------------


def build_graph_dataset(
    data: pd.DataFrame, data_cfg: DataConfig, repr_cfg: RepresentationConfig, out_dir: Path
):
    """
    Write the raw GNN CSV and construct a ``CustomPolymerGraph`` dataset.

    The CSV is saved under ``out_dir/representation/GNN/raw/{data_name}``
    with ``index=False``. When ``id_col`` is present it is placed first so
    that callers can read back the file with ``index_col=0`` to recover the
    sample ID as the DataFrame index.

    Parameters
    ----------
    data:
        Full dataset (SMILES, target, optional id/weight columns).
        May have ``id_col`` either as a regular column or as the index.
    data_cfg:
        Data configuration (``smiles_cols``, ``target_variable_col``,
        ``id_col``, ``data_name``).
    repr_cfg:
        Representation configuration (``node_features``, ``edge_features``,
        ``weights_col``).
    out_dir:
        Experiment root directory.

    Returns
    -------
    CustomPolymerGraph
    """
    from polynet.featurizer.polymer_graph import CustomPolymerGraph

    raw_dir = out_dir / "representation" / "GNN" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Normalise: bring id_col from index into columns if needed
    df = data.copy()
    if data_cfg.id_col and df.index.name == data_cfg.id_col:
        df = df.reset_index()

    keep = []
    if data_cfg.id_col and data_cfg.id_col in df.columns:
        keep.append(data_cfg.id_col)
    keep += data_cfg.smiles_cols + [data_cfg.target_variable_col]
    if repr_cfg.weights_col:
        keep += [c for c in repr_cfg.weights_col.values() if c in df.columns]

    df[keep].to_csv(raw_dir / data_cfg.data_name, index=False)

    dataset = CustomPolymerGraph(
        root=str(raw_dir.parent),
        filename=data_cfg.data_name,
        smiles_cols=data_cfg.smiles_cols,
        weights_col=repr_cfg.weights_col,
        target_col=data_cfg.target_variable_col,
        id_col=data_cfg.id_col,
        node_feats=repr_cfg.node_features,
        edge_feats=repr_cfg.edge_features,
    )
    logger.info(
        f"Graph dataset: {len(dataset)} graphs, "
        f"{dataset[0].num_node_features} node features, "
        f"{dataset[0].num_edge_features} edge features"
    )
    return dataset


def compute_descriptors(
    data: pd.DataFrame, data_cfg: DataConfig, repr_cfg: RepresentationConfig, out_dir: Path
) -> dict:
    """
    Compute molecular descriptors, save sanitised CSVs to disk, and return
    sanitised DataFrames ready for training.

    CSVs are written under ``out_dir/representation/Descriptors/{name}.csv``
    **with** the pandas index so that ``load_dataframes`` can read them back
    correctly with ``index_col=0``. Sanitisation (dropping SMILES and weight
    columns, ensuring the target is last) is applied before saving and before
    returning.

    Parameters
    ----------
    data:
        Full dataset DataFrame.
    data_cfg:
        Data configuration.
    repr_cfg:
        Representation configuration (``molecular_descriptors``,
        ``smiles_merge_approach``, ``weights_col``, etc.).
    out_dir:
        Experiment root directory.

    Returns
    -------
    dict[MolecularDescriptor, pd.DataFrame]
        Sanitised DataFrames keyed by descriptor type.
    """
    from polynet.data.preprocessing import sanitise_df
    from polynet.featurizer.descriptors import build_vector_representation

    representation_dir = out_dir / "representation" / "Descriptors"
    representation_dir.mkdir(parents=True, exist_ok=True)

    weights_cols = list(repr_cfg.weights_col.values()) if repr_cfg.weights_col else None

    desc_dfs = build_vector_representation(
        data=data,
        molecular_descriptors=repr_cfg.molecular_descriptors,
        smiles_cols=data_cfg.smiles_cols,
        id_col=data_cfg.id_col,
        target_col=data_cfg.target_variable_col,
        merging_approach=repr_cfg.smiles_merge_approach,
        weights_col=repr_cfg.weights_col,
        rdkit_independent=repr_cfg.rdkit_independent,
        df_descriptors_independent=repr_cfg.df_descriptors_independent,
        mix_rdkit_df_descriptors=repr_cfg.mix_rdkit_df_descriptors,
    )

    sanitised = {}
    for name, desc_df in desc_dfs.items():
        if desc_df is None:
            continue
        # Save with the pandas index (fixes the previous app index=False bug).
        # load_dataframes reads back with index_col=0 and expects this format.
        desc_df.to_csv(representation_dir / f"{name}.csv")
        sanitised[name] = sanitise_df(
            df=desc_df,
            smiles_cols=data_cfg.smiles_cols,
            target_variable_col=data_cfg.target_variable_col,
            weights_cols=weights_cols,
        )
        logger.info(f"Descriptor set '{name}': {sanitised[name].shape[1] - 1} features")

    return sanitised


# ---------------------------------------------------------------------------
# Splitting stage
# ---------------------------------------------------------------------------


def compute_data_splits(
    data: pd.DataFrame,
    data_cfg: DataConfig,
    split_cfg: SplitConfig,
    random_seed: int,
    out_dir: Path | None = None,
) -> tuple[list, list, list]:
    """
    Compute train/val/test split indices and optionally persist them to disk.

    Parameters
    ----------
    data:
        Dataset DataFrame (used only for length and stratification).
    data_cfg:
        Data configuration (provides ``target_variable_col``).
    split_cfg:
        Splitting strategy, ratios, and bootstrap settings.
    random_seed:
        Global random seed for reproducibility.
    out_dir:
        If provided, writes ``split_indices.json`` to this directory.

    Returns
    -------
    tuple[list, list, list]
        ``(train_idxs, val_idxs, test_idxs)`` — each a list of length
        ``n_bootstrap_iterations``.
    """
    from polynet.factories.dataloader import get_data_split_indices

    train_idxs, val_idxs, test_idxs = get_data_split_indices(
        data=data,
        split_type=split_cfg.split_type,
        split_method=split_cfg.split_method,
        n_bootstrap_iterations=split_cfg.n_bootstrap_iterations,
        val_ratio=split_cfg.val_ratio,
        test_ratio=split_cfg.test_ratio,
        target_variable_col=data_cfg.target_variable_col,
        train_set_balance=split_cfg.train_set_balance,
        random_seed=random_seed,
    )

    for i, (tr, va, te) in enumerate(zip(train_idxs, val_idxs, test_idxs)):
        logger.info(f"Split {i + 1}: train={len(tr)}, val={len(va)}, test={len(te)}")

    if out_dir is not None:
        splits_file = out_dir / "split_indices.json"
        with open(splits_file, "w") as f:
            json.dump(
                {
                    "train": [list(map(str, s)) for s in train_idxs],
                    "val": [list(map(str, s)) for s in val_idxs],
                    "test": [list(map(str, s)) for s in test_idxs],
                },
                f,
                indent=2,
            )
        logger.info(f"Split indices saved to {splits_file}")

    return train_idxs, val_idxs, test_idxs


# ---------------------------------------------------------------------------
# GNN stages
# ---------------------------------------------------------------------------


def train_gnn(
    dataset,
    split_indexes: tuple,
    data_cfg: DataConfig,
    gnn_cfg: TrainGNNConfig,
    random_seed: int,
    out_dir: Path,
) -> tuple[dict, dict]:
    """
    Train a GNN ensemble, save ``.pt`` model files, and return
    ``(trained_models, loaders)``.

    Model files are written to ``out_dir/ml_results/models/``.

    Parameters
    ----------
    dataset:
        ``CustomPolymerGraph`` dataset.
    split_indexes:
        ``(train_idxs, val_idxs, test_idxs)`` from ``compute_data_splits``.
    data_cfg:
        Data configuration (``problem_type``, ``num_classes``).
    gnn_cfg:
        GNN architecture and training configuration.
    random_seed:
        Global random seed.
    out_dir:
        Experiment root directory.

    Returns
    -------
    tuple[dict, dict]
        ``(trained_models, loaders)``
    """
    from torch import save as torch_save

    from polynet.training.gnn import train_gnn_ensemble

    models_dir = out_dir / "ml_results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    trained_models, loaders = train_gnn_ensemble(
        experiment_path=out_dir,
        dataset=dataset,
        split_indexes=split_indexes,
        gnn_conv_params=gnn_cfg.gnn_convolutional_layers,
        problem_type=data_cfg.problem_type,
        num_classes=data_cfg.num_classes,
        random_seed=random_seed,
    )

    for model_name, model in trained_models.items():
        torch_save(model, models_dir / f"{model_name}.pt")

    logger.info(f"Trained GNN models: {list(trained_models.keys())}")
    return trained_models, loaders


def run_gnn_inference(
    trained_models: dict, loaders: dict, data_cfg: DataConfig, split_cfg: SplitConfig
) -> pd.DataFrame:
    """
    Run GNN inference on all splits and return a predictions DataFrame.

    Parameters
    ----------
    trained_models:
        Dict of ``{model_name: model}`` from ``train_gnn``.
    loaders:
        Dict of DataLoaders from ``train_gnn``.
    data_cfg:
        Data configuration (``problem_type``, ``target_variable_name``).
    split_cfg:
        Split configuration (``split_type``).

    Returns
    -------
    pd.DataFrame
    """
    from polynet.inference.gnn import get_predictions_df_gnn

    predictions = get_predictions_df_gnn(
        models=trained_models,
        loaders=loaders,
        problem_type=data_cfg.problem_type,
        split_type=split_cfg.split_type,
        target_variable_name=data_cfg.target_variable_name,
    )
    logger.info(f"GNN predictions shape: {predictions.shape}")
    return predictions


# ---------------------------------------------------------------------------
# TML stages
# ---------------------------------------------------------------------------


def train_tml(
    desc_dfs: dict,
    split_indexes: tuple,
    data_cfg: DataConfig,
    tml_cfg: TrainTMLConfig,
    preprocessing_cfg: FeatureTransformConfig,
    random_seed: int,
    out_dir: Path,
) -> tuple[dict, dict, dict]:
    """
    Train a TML ensemble, save model files, and return
    ``(trained, training_data, scalers)``.

    Model files (``.joblib``) and scaler files (``.pkl``) are written to
    ``out_dir/ml_results/models/``.

    Parameters
    ----------
    desc_dfs:
        Sanitised descriptor DataFrames from ``compute_descriptors`` or
        ``load_dataframes``, keyed by ``MolecularDescriptor``.
    split_indexes:
        ``(train_idxs, val_idxs, test_idxs)`` from ``compute_data_splits``.
    data_cfg:
        Data configuration (``problem_type``).
    tml_cfg:
        TML model selection configuration.
    preprocessing_cfg:
        Feature scaling and selection configuration.
    random_seed:
        Global random seed.
    out_dir:
        Experiment root directory.

    Returns
    -------
    tuple[dict, dict, dict]
        ``(trained, training_data, scalers)``
    """
    import joblib

    from polynet.training.tml import train_tml_ensemble

    models_dir = out_dir / "ml_results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    trained, training_data, scalers = train_tml_ensemble(
        tml_models=tml_cfg.selected_models,
        problem_type=data_cfg.problem_type,
        transform_type=preprocessing_cfg.scaler,
        feature_selection=preprocessing_cfg.selectors,
        dataframes=desc_dfs,
        random_seed=random_seed,
        train_val_test_idxs=split_indexes,
    )

    for model_name, model in trained.items():
        joblib.dump(model, models_dir / f"{model_name}.joblib")
    if scalers:
        for scaler_name, scaler in scalers.items():
            joblib.dump(scaler, models_dir / f"{scaler_name}.pkl")

    logger.info(f"Trained TML models: {list(trained.keys())}")
    return trained, training_data, scalers


def run_tml_inference(
    trained: dict, training_data: dict, data_cfg: DataConfig, split_cfg: SplitConfig
) -> pd.DataFrame:
    """
    Run TML inference on all splits and return a predictions DataFrame.

    Parameters
    ----------
    trained:
        Dict of ``{model_name: model}`` from ``train_tml``.
    training_data:
        Training DataFrames returned by ``train_tml``.
    data_cfg:
        Data configuration.
    split_cfg:
        Split configuration.

    Returns
    -------
    pd.DataFrame
    """
    from polynet.inference.tml import get_predictions_df_tml

    predictions = get_predictions_df_tml(
        models=trained,
        training_data=training_data,
        split_type=split_cfg.split_type,
        target_variable_col=data_cfg.target_variable_col,
        problem_type=data_cfg.problem_type,
        target_variable_name=data_cfg.target_variable_name,
    )
    logger.info(f"TML predictions shape: {predictions.shape}")
    return predictions


# ---------------------------------------------------------------------------
# Evaluation stages
# ---------------------------------------------------------------------------


def compute_metrics(
    predictions: pd.DataFrame, trained_models: dict, data_cfg: DataConfig, split_cfg: SplitConfig
) -> dict:
    """
    Compute evaluation metrics for a predictions DataFrame.

    Parameters
    ----------
    predictions:
        Output of ``run_gnn_inference`` or ``run_tml_inference``.
    trained_models:
        Trained model dict (used to enumerate model names).
    data_cfg:
        Data configuration (``problem_type``, ``target_variable_name``).
    split_cfg:
        Split configuration (``split_type``).

    Returns
    -------
    dict
        Nested dict ``{iteration: {model: {set: {metric: value}}}}``.
    """
    from polynet.training.metrics import get_metrics

    return get_metrics(
        predictions=predictions,
        split_type=split_cfg.split_type,
        target_variable_name=data_cfg.target_variable_name,
        trained_models=list(trained_models.keys()),
        problem_type=data_cfg.problem_type,
    )


def plot_results_stage(
    predictions: pd.DataFrame,
    trained_models: dict,
    data_cfg: DataConfig,
    split_cfg: SplitConfig,
    plots_dir: Path,
) -> None:
    """
    Generate learning curves and result plots, saving them to ``plots_dir``.

    Parameters
    ----------
    predictions:
        Output of ``run_gnn_inference`` or ``run_tml_inference``.
    trained_models:
        Trained model dict.
    data_cfg:
        Data configuration (``problem_type``, ``target_variable_name``,
        ``class_names``).
    split_cfg:
        Split configuration (``split_type``).
    plots_dir:
        Directory where plots are saved. Created if it does not exist.
    """
    from polynet.training.evaluate import plot_learning_curves, plot_results

    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_learning_curves(models=trained_models, save_path=plots_dir)

    class_names = data_cfg.class_names
    if isinstance(class_names, dict):
        class_names = {int(k): v for k, v in class_names.items()}

    plot_results(
        predictions=predictions,
        split_type=split_cfg.split_type,
        target_variable_name=data_cfg.target_variable_name,
        ml_algorithms=list(trained_models.keys()),
        problem_type=data_cfg.problem_type,
        save_path=plots_dir,
        class_names=class_names,
    )
    logger.info(f"Plots saved to {plots_dir}")


# ---------------------------------------------------------------------------
# External prediction stage
# ---------------------------------------------------------------------------


def predict_external(
    data: pd.DataFrame,
    data_cfg: DataConfig,
    repr_cfg: RepresentationConfig,
    experiment_path: Path,
    out_dir: Path,
    dataset_name: str,
) -> tuple[pd.DataFrame, dict | None]:
    """
    Predict properties for an external (unseen) dataset using models from a
    trained experiment.

    Both GNN and TML models found under ``experiment_path/ml_results/models/``
    are loaded automatically.  If the target column is present in ``data``,
    per-model metrics are computed and returned; otherwise metrics are ``None``.

    Parameters
    ----------
    data:
        Unseen DataFrame.  Must contain the SMILES column(s) and, optionally,
        the target variable column and weight fraction columns.  The id column
        may be a regular column or the DataFrame index.
    data_cfg:
        DataConfig from the training experiment.
    repr_cfg:
        RepresentationConfig from the training experiment.
    experiment_path:
        Root directory of the trained experiment (models are loaded from
        ``experiment_path/ml_results/models/``).
    out_dir:
        Directory where ``predictions.csv`` (and ``metrics.json`` when the
        target is present) will be written.  Created if it does not exist.
    dataset_name:
        File name of the unseen dataset (e.g. ``"test_set.csv"``).  Used as
        the graph dataset filename so the GNN featuriser saves raw data under
        ``out_dir/representation/GNN/raw/``.

    Returns
    -------
    tuple[pd.DataFrame, dict | None]
        ``(predictions, metrics)`` — metrics is ``None`` when the target column
        is absent from ``data``.
    """
    import json

    import joblib
    import torch
    from torch_geometric.loader import DataLoader

    from polynet.config.column_names import (
        get_predicted_label_column_name,
        get_true_label_column_name,
    )
    from polynet.config.constants import ResultColumn
    from polynet.config.enums import ProblemType
    from polynet.data.preprocessing import sanitise_df
    from polynet.featurizer.descriptors import build_vector_representation
    from polynet.featurizer.polymer_graph import CustomPolymerGraph
    from polynet.inference.utils import prepare_probs_df
    from polynet.training.metrics import calculate_metrics

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Normalise: bring id_col from index into columns if needed
    # ------------------------------------------------------------------
    df = data.copy()
    if data_cfg.id_col and df.index.name == data_cfg.id_col:
        df = df.reset_index()

    has_target = data_cfg.target_variable_col in df.columns

    # ------------------------------------------------------------------
    # Build id_df — carries index, SMILES, weights, optional true label
    # ------------------------------------------------------------------
    id_cols = []
    if data_cfg.id_col and data_cfg.id_col in df.columns:
        id_cols.append(data_cfg.id_col)
    id_cols += [c for c in data_cfg.smiles_cols if c in df.columns]
    if repr_cfg.weights_col:
        id_cols += [c for c in repr_cfg.weights_col.values() if c in df.columns]
    if has_target:
        id_cols.append(data_cfg.target_variable_col)

    id_df = df[id_cols].copy().reset_index(drop=True)
    true_label_name = get_true_label_column_name(data_cfg.target_variable_name)
    id_df = id_df.rename(
        columns={data_cfg.id_col: ResultColumn.INDEX, data_cfg.target_variable_col: true_label_name}
    )

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    models_dir = experiment_path / "ml_results" / "models"
    gnn_model_files = sorted(models_dir.glob("*.pt"))
    tml_model_files = sorted(models_dir.glob("*.joblib"))

    gnn_models = {
        f.stem: torch.load(f, weights_only=False, map_location="cpu") for f in gnn_model_files
    }
    tml_models = {f.stem: joblib.load(f) for f in tml_model_files}

    # Scalers: one .pkl per descriptor type, named by the last '-' segment of
    # the model stem (e.g. "rf-Morgan_1" → "Morgan_1.pkl").
    scalers: dict = {}
    seen_scaler_names: set = set()
    for f in tml_model_files:
        scaler_name = f.stem.rsplit("-", 1)[-1]
        scaler_path = models_dir / f"{scaler_name}.pkl"
        if scaler_name not in seen_scaler_names and scaler_path.exists():
            scalers[scaler_name] = joblib.load(scaler_path)
            seen_scaler_names.add(scaler_name)

    logger.info(f"Loaded {len(gnn_models)} GNN model(s), {len(tml_models)} TML model(s)")

    predictions_tml: pd.DataFrame | None = None
    predictions_gnn: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # TML path: descriptors → predict
    # ------------------------------------------------------------------
    if tml_models:
        weights_cols = list(repr_cfg.weights_col.values()) if repr_cfg.weights_col else None
        raw_desc_dfs = build_vector_representation(
            data=df,
            molecular_descriptors=repr_cfg.molecular_descriptors,
            smiles_cols=data_cfg.smiles_cols,
            id_col=data_cfg.id_col,
            target_col=data_cfg.target_variable_col,
            merging_approach=repr_cfg.smiles_merge_approach,
            weights_col=repr_cfg.weights_col,
            rdkit_independent=repr_cfg.rdkit_independent,
            df_descriptors_independent=repr_cfg.df_descriptors_independent,
            mix_rdkit_df_descriptors=repr_cfg.mix_rdkit_df_descriptors,
        )
        descriptor_dfs = {
            name: sanitise_df(
                df=desc_df,
                smiles_cols=data_cfg.smiles_cols,
                target_variable_col=data_cfg.target_variable_col,
                weights_cols=weights_cols,
            )
            for name, desc_df in raw_desc_dfs.items()
            if desc_df is not None
        }

        preds_all = None
        for model_name, model in tml_models.items():
            model_log_name = model_name.replace("_", " ")
            predicted_col = get_predicted_label_column_name(
                target_variable_name=data_cfg.target_variable_name, model_name=model_log_name
            )
            ml_model = model_name.rsplit("_", 1)[0]  # e.g. "rf-Morgan"
            df_name = ml_model.rsplit("-", 1)[1]  # e.g. "Morgan"
            desc_df = descriptor_dfs[df_name]

            if scalers:
                scaler_name = model_name.rsplit("-", 1)[-1]
                scaler = scalers[scaler_name]
                desc_arr = scaler.transform(desc_df)
                desc_df = pd.DataFrame(desc_arr, columns=scaler.get_feature_names_out())

            preds = model.predict(desc_df)
            preds_df = pd.DataFrame({predicted_col: preds})

            if data_cfg.problem_type == ProblemType.Classification:
                probs_df = prepare_probs_df(
                    probs=model.predict_proba(desc_df),
                    target_variable_name=data_cfg.target_variable_name,
                    model_name=model_log_name,
                )
                preds_df[probs_df.columns] = probs_df.to_numpy()

            preds_all = preds_df if preds_all is None else pd.concat([preds_all, preds_df], axis=1)

        if preds_all is not None:
            predictions_tml = pd.concat([id_df, preds_all], axis=1, ignore_index=False)

    # ------------------------------------------------------------------
    # GNN path: graph dataset → predict → ensemble voting
    # ------------------------------------------------------------------
    if gnn_models:
        raw_gnn_dir = out_dir / "representation" / "GNN" / "raw"
        raw_gnn_dir.mkdir(parents=True, exist_ok=True)

        keep = []
        if data_cfg.id_col and data_cfg.id_col in df.columns:
            keep.append(data_cfg.id_col)
        keep += data_cfg.smiles_cols + [data_cfg.target_variable_col]
        if repr_cfg.weights_col:
            keep += [c for c in repr_cfg.weights_col.values() if c in df.columns]
        df[keep].to_csv(raw_gnn_dir / dataset_name, index=False)

        dataset = CustomPolymerGraph(
            filename=dataset_name,
            root=str(raw_gnn_dir.parent),
            smiles_cols=data_cfg.smiles_cols,
            target_col=data_cfg.target_variable_col,
            id_col=data_cfg.id_col,
            weights_col=repr_cfg.weights_col,
            node_feats=repr_cfg.node_features,
            edge_feats=repr_cfg.edge_features,
        )

        preds_all = None
        for model_name, model in gnn_models.items():
            model_log_name = model_name.replace("_", " ")
            predicted_col = get_predicted_label_column_name(
                target_variable_name=data_cfg.target_variable_name, model_name=model_log_name
            )
            loader = DataLoader(dataset)
            preds = model.predict_loader(loader)
            preds_df = pd.DataFrame({ResultColumn.INDEX: preds[0], predicted_col: preds[1]})

            if data_cfg.problem_type == ProblemType.Classification:
                probs_df = prepare_probs_df(
                    probs=preds[-1],
                    target_variable_name=data_cfg.target_variable_name,
                    model_name=model_log_name,
                )
                preds_df[probs_df.columns] = probs_df.to_numpy()

            preds_all = (
                preds_df
                if preds_all is None
                else pd.merge(preds_all, preds_df, on=[ResultColumn.INDEX])
            )

        if preds_all is not None:
            # Ensemble voting per architecture
            pred_cols = [c for c in preds_all.columns if ResultColumn.PREDICTED in c]
            arch_groups: dict[str, list[str]] = {}
            for col in pred_cols:
                arch = col.split(" ")[0]
                arch_groups.setdefault(arch, []).append(col)
            arch_groups["GNN"] = pred_cols  # all GNN models together

            ensemble_series = []
            for arch, cols in arch_groups.items():
                if len(cols) < 2:
                    continue
                arch_preds = preds_all[cols]
                if data_cfg.problem_type == ProblemType.Classification:
                    from scipy.stats import mode as scipy_mode

                    votes, _ = scipy_mode(arch_preds.values, axis=1, keepdims=False)
                    ensemble_series.append(
                        pd.Series(
                            votes,
                            index=arch_preds.index,
                            name=f"{arch} Ensemble {ResultColumn.PREDICTED}",
                        )
                    )
                else:
                    ensemble_series.append(
                        pd.Series(
                            arch_preds.mean(axis=1),
                            index=arch_preds.index,
                            name=f"{arch} Ensemble {ResultColumn.PREDICTED}",
                        )
                    )
            if ensemble_series:
                preds_all = pd.concat([preds_all] + ensemble_series, axis=1)

            predictions_gnn = pd.merge(id_df, preds_all, on=[ResultColumn.INDEX])

    # ------------------------------------------------------------------
    # Merge TML + GNN predictions
    # ------------------------------------------------------------------
    if predictions_tml is not None and predictions_gnn is not None:
        predictions = pd.merge(left=predictions_tml, right=predictions_gnn, on=list(id_df.columns))
    elif predictions_gnn is not None:
        predictions = predictions_gnn
    elif predictions_tml is not None:
        predictions = predictions_tml
    else:
        raise RuntimeError(
            f"No trained models found in {models_dir}. " "Run the training pipeline first."
        )

    # ------------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------------
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    logger.info(f"Predictions saved to {out_dir / 'predictions.csv'}")

    # ------------------------------------------------------------------
    # Metrics (only when true labels are available)
    # ------------------------------------------------------------------
    metrics: dict | None = None
    if has_target:
        label_col = true_label_name
        metrics = {}
        for col in predictions.columns:
            if ResultColumn.PREDICTED not in col or "Ensemble" in col:
                continue
            split_name = col.rsplit(" ", 3)
            model, number = split_name[0], split_name[1]
            model_name_key = f"{model} {number}"
            probs_cols = [
                c for c in predictions.columns if ResultColumn.SCORE in c and model_name_key in c
            ]
            metrics.setdefault(number, {}).setdefault(model, {})[dataset_name.split(".")[0]] = (
                calculate_metrics(
                    y_true=predictions[label_col],
                    y_pred=predictions[col],
                    y_probs=predictions[probs_cols] if probs_cols else None,
                    problem_type=data_cfg.problem_type,
                )
            )

        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {out_dir / 'metrics.json'}")

    return predictions, metrics


# ---------------------------------------------------------------------------
# Explainability stage
# ---------------------------------------------------------------------------


def run_explainability(
    trained_models: dict,
    dataset,
    split_indexes: tuple,
    data_cfg: DataConfig,
    exp_cfg: dict,
    out_dir: Path,
) -> None:
    """
    Run attribution-based explainability and save heatmaps to disk.

    Parameters
    ----------
    trained_models:
        Trained GNN models from ``train_gnn``.
    dataset:
        ``CustomPolymerGraph`` dataset from ``build_graph_dataset``.
    split_indexes:
        ``(train_idxs, val_idxs, test_idxs)`` from ``compute_data_splits``.
    data_cfg:
        Data configuration.
    exp_cfg:
        Raw explainability configuration dict with keys: ``algorithm``,
        ``fragmentation``, ``normalisation``, ``cutoff``,
        ``explain_mol_ids`` (optional).
    out_dir:
        Experiment root directory. Outputs written to
        ``out_dir/explanations/``.
    """
    from polynet.config.enums import (
        ExplainAlgorithm,
        FragmentationMethod,
        ImportanceNormalisationMethod,
    )
    from polynet.explainability.pipeline import run_explanation
    from polynet.explainability.visualization import (
        plot_attribution_distribution,
        plot_mols_with_weights,
    )
    from polynet.visualization.utils import save_plot

    explain_dir = out_dir / "explanations"
    explain_dir.mkdir(parents=True, exist_ok=True)

    explain_mol_ids = exp_cfg.get("explain_mol_ids")
    if explain_mol_ids is None:
        _, _, test_idxs = split_indexes
        explain_mol_ids = [str(idx) for idx in test_idxs[0][:5]]
        logger.info(f"explain_mol_ids not set — using first 5 test samples: {explain_mol_ids}")

    result = run_explanation(
        models=trained_models,
        dataset=dataset,
        explain_mol_ids=explain_mol_ids,
        plot_mol_ids=explain_mol_ids,
        algorithm=ExplainAlgorithm(exp_cfg["algorithm"]),
        problem_type=data_cfg.problem_type,
        experiment_path=out_dir,
        node_features=dataset.node_feats,
        normalisation=ImportanceNormalisationMethod(exp_cfg.get("normalisation", "Local")),
        cutoff=exp_cfg.get("cutoff", 0.05),
        fragmentation_method=FragmentationMethod(exp_cfg.get("fragmentation", "brics")),
    )

    fig = plot_attribution_distribution(result.fragment_importances)
    save_plot(fig, explain_dir / "fragment_attributions.png")
    logger.info("Saved fragment_attributions.png")

    for mol_exp in result.mol_explanations:
        fig = plot_mols_with_weights(
            smiles_list=mol_exp.monomer_smiles,
            weights_list=mol_exp.per_monomer_weights,
            legend=mol_exp.monomer_smiles,
        )
        save_plot(fig, explain_dir / f"{mol_exp.mol_id}_heatmap.png")
        logger.info(f"Saved {mol_exp.mol_id}_heatmap.png")
