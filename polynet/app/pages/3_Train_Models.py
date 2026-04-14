import json
from shutil import rmtree

import pandas as pd
import streamlit as st

from polynet.app.components.experiments import experiment_selector
from polynet.app.components.forms.train_models import (
    split_data_form,
    target_transform_widget,
    train_GNN_models_form,
    train_TML_models,
)
from polynet.app.components.plots import display_model_results
from polynet.app.options.file_paths import (
    data_options_path,
    data_spliting_options_path,
    general_options_path,
    gnn_raw_data_file,
    gnn_raw_data_path,
    ml_results_file_path,
    ml_results_parent_directory,
    model_metrics_file_path,
    plots_directory,
    polynet_experiment_path,
    preprocessing_tml_model_options_path,
    representation_file_path,
    representation_options_path,
    target_transform_options_path,
    train_gnn_model_options_path,
    train_tml_model_options_path,
)
from polynet.app.options.state_keys import (
    GeneralConfigStateKeys,
    TrainGNNStateKeys,
    TrainTMLStateKeys,
)
from polynet.app.services.configurations import load_options, save_options
from polynet.app.services.experiments import get_experiments
from polynet.app.services.model_training import load_dataframes
from polynet.app.utils import save_data
from polynet.config.column_names import get_iterator_name, get_true_label_column_name
from polynet.config.constants import ResultColumn
from polynet.config.schemas import (
    DataConfig,
    FeatureTransformConfig,
    GeneralConfig,
    RepresentationConfig,
    SplitConfig,
    TargetTransformConfig,
    TrainGNNConfig,
    TrainTMLConfig,
)
from polynet.featurizer.polymer_graph import CustomPolymerGraph
from polynet.pipeline import (
    compute_data_splits,
    compute_metrics,
    plot_results_stage,
    run_gnn_inference,
    run_tml_inference,
    train_gnn,
    train_tml,
)


def train_models(
    experiment_name: str,
    tml_models: dict,
    preprocessing_cfg: FeatureTransformConfig,
    gnn_conv_params: dict,
    representation_options: RepresentationConfig,
    data_options: DataConfig,
    target_cfg: TargetTransformConfig | None = None,
):

    if target_cfg is None:
        target_cfg = TargetTransformConfig()

    # paths for options and experiments
    experiment_path = polynet_experiment_path(experiment_name=experiment_name)
    tml_training_opts_path = train_tml_model_options_path(experiment_path=experiment_path)
    preprocessing_opts_path = preprocessing_tml_model_options_path(experiment_path=experiment_path)
    gnn_training_opts_path = train_gnn_model_options_path(experiment_path=experiment_path)
    target_transform_opts_path = target_transform_options_path(experiment_path=experiment_path)
    ml_results_dir = ml_results_parent_directory(experiment_path=experiment_path)
    split_cfg_path = data_spliting_options_path(experiment_path=experiment_path)
    gen_options_path = general_options_path(experiment_path=experiment_path)

    # delete old options if they exist
    if tml_training_opts_path.exists():
        tml_training_opts_path.unlink()
    if preprocessing_opts_path.exists():
        preprocessing_opts_path.unlink()
    if gnn_training_opts_path.exists():
        gnn_training_opts_path.unlink()
    if target_transform_opts_path.exists():
        target_transform_opts_path.unlink()
    if ml_results_dir.exists():
        rmtree(ml_results_dir)
    if split_cfg_path.exists():
        split_cfg_path.unlink()
    # Load general experiment options saved by Page 1 (Create Experiment)
    general_experiment_options = load_options(path=gen_options_path, options_class=GeneralConfig)

    split_cfg = SplitConfig(
        split_type=st.session_state[GeneralConfigStateKeys.SplitType],
        split_method=st.session_state[GeneralConfigStateKeys.SplitMethod],
        train_set_balance=st.session_state.get(GeneralConfigStateKeys.DesiredProportion, None),
        test_ratio=st.session_state[GeneralConfigStateKeys.TestSize],
        val_ratio=st.session_state[GeneralConfigStateKeys.ValidationSize],
        n_bootstrap_iterations=st.session_state.get(GeneralConfigStateKeys.BootstrapIterations, 1),
    )
    save_options(split_cfg_path, split_cfg)
    save_options(target_transform_opts_path, target_cfg)

    # read the data
    data = pd.read_csv(
        gnn_raw_data_file(file_name=data_options.data_name, experiment_path=experiment_path),
        index_col=0,
    )

    # Compute data splits using the shared pipeline stage
    train_val_test_idxs = compute_data_splits(
        data=data,
        data_cfg=data_options,
        split_cfg=split_cfg,
        random_seed=general_experiment_options.random_seed,
        out_dir=experiment_path,
    )

    # Create directory to save plots
    plots_dir = plots_directory(experiment_path=experiment_path)
    plots_dir.mkdir(parents=True)

    metrics_path = model_metrics_file_path(experiment_path=experiment_path)

    # ------------------------------------------------------------------
    # TML training
    # ------------------------------------------------------------------
    if tml_models:
        tml_cfg = TrainTMLConfig(
            train_tml=st.session_state[TrainTMLStateKeys.TrainTML], selected_models=tml_models
        )
        save_options(path=tml_training_opts_path, options=tml_cfg)
        save_options(path=preprocessing_opts_path, options=preprocessing_cfg)

        # load descriptor DataFrames from disk (saved by Page 2)
        dataframes = load_dataframes(
            representation_options=representation_options,
            data_options=data_options,
            experiment_path=experiment_path,
        )

        # train via shared stage (also saves .joblib/.pkl model files)
        tml_trained, tml_training_data, _, tml_target_scalers = train_tml(
            desc_dfs=dataframes,
            split_indexes=train_val_test_idxs,
            data_cfg=data_options,
            tml_cfg=tml_cfg,
            preprocessing_cfg=preprocessing_cfg,
            random_seed=general_experiment_options.random_seed,
            out_dir=experiment_path,
            target_cfg=target_cfg,
        )

        # generate predictions DataFrame (inverse-transform via target_scalers)
        tml_predictions_df = run_tml_inference(
            trained=tml_trained,
            training_data=tml_training_data,
            data_cfg=data_options,
            split_cfg=split_cfg,
            target_scalers=tml_target_scalers,
        )

        metrics_tml = compute_metrics(
            predictions=tml_predictions_df,
            trained_models=tml_trained,
            data_cfg=data_options,
            split_cfg=split_cfg,
        )

        plot_results_stage(
            predictions=tml_predictions_df,
            trained_models=tml_trained,
            data_cfg=data_options,
            split_cfg=split_cfg,
            plots_dir=plots_dir,
        )

    # ------------------------------------------------------------------
    # GNN training
    # ------------------------------------------------------------------
    if gnn_conv_params:
        gnn_cfg = TrainGNNConfig(
            train_gnn=st.session_state[TrainGNNStateKeys.TrainGNN],
            gnn_convolutional_layers=gnn_conv_params,
            share_gnn_parameters=st.session_state.get(TrainGNNStateKeys.SharedGNNParams, False),
        )
        save_options(path=gnn_training_opts_path, options=gnn_cfg)

        dataset = CustomPolymerGraph(
            filename=data_options.data_name,
            root=gnn_raw_data_path(experiment_path).parent,
            smiles_cols=data_options.smiles_cols,
            target_col=data_options.target_variable_col,
            id_col=data_options.id_col,
            weights_col=representation_options.weights_col,
            node_feats=representation_options.node_features,
            edge_feats=representation_options.edge_features,
            polymer_descriptors=representation_options.polymer_descriptors,
        )

        # train via shared stage (also saves .pt model files)
        gnn_trained, gnn_loaders, gnn_target_scalers = train_gnn(
            dataset=dataset,
            split_indexes=train_val_test_idxs,
            data_cfg=data_options,
            gnn_cfg=gnn_cfg,
            random_seed=general_experiment_options.random_seed,
            out_dir=experiment_path,
            target_cfg=target_cfg,
        )

        gnn_predictions_df = run_gnn_inference(
            trained_models=gnn_trained,
            loaders=gnn_loaders,
            data_cfg=data_options,
            split_cfg=split_cfg,
            target_scalers=gnn_target_scalers,
        )

        metrics_gnn = compute_metrics(
            predictions=gnn_predictions_df,
            trained_models=gnn_trained,
            data_cfg=data_options,
            split_cfg=split_cfg,
        )

        plot_results_stage(
            predictions=gnn_predictions_df,
            trained_models=gnn_trained,
            data_cfg=data_options,
            split_cfg=split_cfg,
            plots_dir=plots_dir,
        )

    # ------------------------------------------------------------------
    # Merge predictions and metrics when both model families are trained
    # ------------------------------------------------------------------
    if tml_models and gnn_conv_params:
        iterator = get_iterator_name(split_cfg.split_type)
        label_col_name = get_true_label_column_name(
            target_variable_name=data_options.target_variable_name
        )
        gnn_predictions_df = gnn_predictions_df.drop(columns=[label_col_name])

        predictions = pd.merge(
            left=tml_predictions_df,
            right=gnn_predictions_df,
            on=[ResultColumn.INDEX, ResultColumn.SET, iterator],
        )

        metrics = {}
        for i in range(split_cfg.n_bootstrap_iterations):
            iteration = str(i + 1)
            metrics[iteration] = {**metrics_gnn[iteration], **metrics_tml[iteration]}

    elif gnn_conv_params:
        predictions = gnn_predictions_df.copy()
        metrics = metrics_gnn
    elif tml_models:
        predictions = tml_predictions_df.copy()
        metrics = metrics_tml

    save_data(data=predictions, data_path=ml_results_file_path(experiment_path=experiment_path))

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    display_model_results(experiment_path=experiment_path, expanded=True)


st.header("Train Models")

st.markdown(
    """
    In this section, you can train machine learning models on your polymer dataset. The models will be trained using the representations you built in the previous section. You can choose from various machine learning algorithms, including Graph Neural Networks (GNNs), Random Forests, and Support Vector Machines (SVMs).
    If you have not yet built a representation of your polymers, please go back to the previous section and build a representation first. The models will be trained on the representations you built, so it is essential to have a representation before training models.
    """
)


choices = get_experiments()
experiment_name = experiment_selector(choices)


if experiment_name:

    experiment_path = polynet_experiment_path(experiment_name=experiment_name)

    path_to_data_opts = data_options_path(experiment_path=experiment_path)

    data_opts = load_options(path=path_to_data_opts, options_class=DataConfig)

    path_to_representation_opts = representation_options_path(experiment_path=experiment_path)
    representation_opts = load_options(
        path=path_to_representation_opts, options_class=RepresentationConfig
    )

    train_gnn_options = train_gnn_model_options_path(experiment_path=experiment_path)

    if train_gnn_options.exists():
        st.error(
            "GNN model options already exist for this experiment. "
            "You can modify the settings below, but be aware that this will overwrite the existing results."
        )
        display_model_results(experiment_path=experiment_path, expanded=False)

    st.markdown("## Train Machine Learning Models (TMLs)")

    if representation_file_path(experiment_path=experiment_path).exists():

        tml_models, preprocessing_cfg = train_TML_models(problem_type=data_opts.problem_type)

    else:
        st.error("No descriptors representation found, TML models cannot be trained.")
        tml_models = {}
        preprocessing_cfg = {}

    st.markdown("## Graph Neural Networks (GNNs)")

    if gnn_raw_data_path(experiment_path=experiment_path).exists():

        gnn_conv_params = train_GNN_models_form(
            representation_opts=representation_opts, problem_type=data_opts.problem_type
        )

    else:
        st.error(
            "No graph representation found. Please build a graph representation of your polymers first."
        )
        gnn_conv_params = {}

    st.markdown("## Data Splitting Options")
    split_data_form(problem_type=data_opts.problem_type)

    # ------------------------------------------------------------------
    # Target variable scaling (regression only)
    # ------------------------------------------------------------------
    target_cfg = TargetTransformConfig()
    if data_opts.problem_type.lower() == "regression":
        st.markdown("## Target Variable Scaling")
        target_cfg = target_transform_widget()

    if gnn_conv_params or tml_models:
        disabled = False
    else:
        disabled = True

    if st.button("Run Training", disabled=disabled):
        st.write("Training models...")

        train_models(
            experiment_name=experiment_name,
            tml_models=tml_models,
            preprocessing_cfg=preprocessing_cfg,
            gnn_conv_params=gnn_conv_params,
            representation_options=representation_opts,
            data_options=data_opts,
            target_cfg=target_cfg,
        )

        st.success("Models trained successfully!")
