import json
import logging

import pandas as pd
import streamlit as st

from polynet.app.components.experiments import experiment_selector
from polynet.app.components.forms.train_models import (
    default_feature_preprocessing,
    feature_transformer_widgets,
    gnn_optimisation_widgets,
    sampling_fingerprint_from_state,
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
    hpo_search_spaces_path,
    ml_results_file_path,
    model_metrics_file_path,
    plots_directory,
    polynet_experiment_path,
    preprocessing_tml_model_options_path,
    representation_file_path,
    representation_options_path,
    split_indices_path,
    target_transform_options_path,
    train_gnn_model_options_path,
    train_tml_model_options_path,
)
from polynet.app.options.state_keys import (
    GeneralConfigStateKeys,
    TrainGNNStateKeys,
    TrainTMLStateKeys,
)
from polynet.app.services.cache import read_csv
from polynet.app.services.configurations import load_options, save_options
from polynet.app.services.experiments import get_experiments
from polynet.app.services.model_training import load_dataframes
from polynet.app.services.progress import run_progress
from polynet.app.services.reruns import clear_stage_results, rerun_warning, stage_rerun_plan
from polynet.app.services.workload import estimate_workload
from polynet.app.utils import save_data
from polynet.config.column_names import get_iterator_name, get_true_label_column_name
from polynet.config.constants import ResultColumn
from polynet.config.enums import SplitSampler
from polynet.config.schemas import (
    DataConfig,
    FeatureTransformConfig,
    GeneralConfig,
    GNNOptimisationConfig,
    RepresentationConfig,
    SplitConfig,
    TargetTransformConfig,
    TrainGNNConfig,
    TrainTMLConfig,
)
from polynet.config.schemas.base import DEFAULT_HPO_NUM_SAMPLES
from polynet.config.search_grid import effective_search_spaces
from polynet.data.sampling import sampling_metadata
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
from polynet.utils.validation import validate_hpo_folds


@run_progress("Training")
def train_models(
    experiment_name: str,
    tml_models: dict,
    preprocessing_cfg: FeatureTransformConfig | None,
    gnn_conv_params: dict,
    representation_options: RepresentationConfig,
    data_options: DataConfig,
    target_cfg: TargetTransformConfig | None = None,
    gnn_optimisation_cfg: GNNOptimisationConfig | None = None,
):

    logger = logging.getLogger("polynet.app.training")
    logger.info("Preparing experiment %s and data splits", experiment_name)
    if target_cfg is None:
        target_cfg = TargetTransformConfig()

    # paths for options and experiments
    experiment_path = polynet_experiment_path(experiment_name=experiment_name)
    tml_training_opts_path = train_tml_model_options_path(experiment_path=experiment_path)
    preprocessing_opts_path = preprocessing_tml_model_options_path(experiment_path=experiment_path)
    gnn_training_opts_path = train_gnn_model_options_path(experiment_path=experiment_path)
    target_transform_opts_path = target_transform_options_path(experiment_path=experiment_path)
    split_cfg_path = data_spliting_options_path(experiment_path=experiment_path)
    gen_options_path = general_options_path(experiment_path=experiment_path)

    # Load general experiment options saved by Page 1 (Create Experiment)
    general_experiment_options = load_options(path=gen_options_path, options_class=GeneralConfig)

    split_cfg = SplitConfig(
        split_type=st.session_state[GeneralConfigStateKeys.SplitType],
        split_method=st.session_state[GeneralConfigStateKeys.SplitMethod],
        train_set_balance=st.session_state.get(GeneralConfigStateKeys.DesiredProportion, None),
        test_ratio=st.session_state[GeneralConfigStateKeys.TestSize],
        val_ratio=st.session_state[GeneralConfigStateKeys.ValidationSize],
        n_bootstrap_iterations=st.session_state.get(GeneralConfigStateKeys.BootstrapIterations, 1),
        sampler=st.session_state.get(GeneralConfigStateKeys.Sampler, SplitSampler.Random),
        sampling_fingerprint=sampling_fingerprint_from_state(),
    )
    # read the data
    raw_path = gnn_raw_data_file(file_name=data_options.data_name, experiment_path=experiment_path)
    data = read_csv(raw_path if raw_path.exists() else data_options.data_path, index_col=0)

    # Compute data splits using the shared pipeline stage. Some samplers cannot
    # split some datasets (e.g. clusters larger than the validation set); show
    # why instead of a traceback.
    try:
        train_val_test_idxs = compute_data_splits(
            data=data,
            data_cfg=data_options,
            split_cfg=split_cfg,
            random_seed=general_experiment_options.random_seed,
            out_dir=None,
            weights_col=representation_options.weights_col,
        )
    except ValueError as e:
        st.error(
            f"The data could not be split with these settings: {e} Try another sampler, "
            "the 'random' split method or larger validation/test ratios."
        )
        st.stop()

    tml_cfg = (
        TrainTMLConfig(
            train_tml=st.session_state[TrainTMLStateKeys.TrainTML],
            selected_models=tml_models,
            hpo_n_folds=int(st.session_state.get(TrainTMLStateKeys.HPONumFolds, 5)),
            include_validation_in_training=st.session_state.get(
                TrainTMLStateKeys.IncludeValidation, True
            ),
            hyperparameter_optimisation=bool(
                st.session_state.get(TrainTMLStateKeys.PerformHyperparameterTuning, False)
            ),
            hpo_num_samples=int(
                st.session_state.get(TrainTMLStateKeys.HPONumSamples, DEFAULT_HPO_NUM_SAMPLES)
            ),
            hpo_search_grid=st.session_state.get(TrainTMLStateKeys.SearchGrid, {}),
        )
        if tml_models
        else None
    )
    gnn_cfg = (
        TrainGNNConfig(
            train_gnn=st.session_state[TrainGNNStateKeys.TrainGNN],
            gnn_convolutional_layers=gnn_conv_params,
            share_gnn_parameters=st.session_state.get(TrainGNNStateKeys.SharedGNNParams, False),
            epochs=int(
                st.session_state.get(
                    TrainGNNStateKeys.Epochs, TrainGNNConfig.model_fields["epochs"].default
                )
            ),
            hyperparameter_optimisation=bool(
                st.session_state.get(TrainGNNStateKeys.HypTunning, False)
            ),
            hpo_num_samples=int(
                st.session_state.get(TrainGNNStateKeys.HPONumSamples, DEFAULT_HPO_NUM_SAMPLES)
            ),
            hpo_search_grid=st.session_state.get(TrainGNNStateKeys.SearchGrid, {}),
            optimisation=gnn_optimisation_cfg or GNNOptimisationConfig(),
        )
        if gnn_conv_params
        else None
    )

    # Check the HPO fold count against the data before any training starts.
    try:
        validate_hpo_folds(
            data=data,
            data_cfg=data_options,
            split_indexes=train_val_test_idxs,
            tml_cfg=tml_cfg,
            gnn_cfg=gnn_cfg,
        )
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # Validation succeeded. Clear stale dependants before writing this run's files.
    try:
        clear_stage_results(experiment_path, "training")
    except OSError as exc:
        st.error(f"Could not clear previous results: {exc}")
        st.stop()
    save_options(split_cfg_path, split_cfg)
    save_options(target_transform_opts_path, target_cfg)
    if preprocessing_cfg is not None:
        save_options(preprocessing_opts_path, preprocessing_cfg)
    save_options(
        split_indices_path(experiment_path),
        {
            **{
                name: [list(map(str, ids)) for ids in groups]
                for name, groups in zip(("train", "val", "test"), train_val_test_idxs)
            },
            "sampling": sampling_metadata(split_cfg, representation_options.weights_col),
        },
    )

    # Provenance: the HPO grids actually searched (default candidates, with any
    # parameter set in hpo_search_grid replacing its defaults).
    search_spaces = effective_search_spaces(
        data_options.problem_type, gnn_cfg=gnn_cfg, tml_cfg=tml_cfg
    )
    if search_spaces:
        save_options(hpo_search_spaces_path(experiment_path), search_spaces)

    # Create directory to save plots
    plots_dir = plots_directory(experiment_path=experiment_path)
    plots_dir.mkdir(parents=True)

    metrics_path = model_metrics_file_path(experiment_path=experiment_path)

    # ------------------------------------------------------------------
    # TML training
    # ------------------------------------------------------------------
    if tml_models:
        save_options(path=tml_training_opts_path, options=tml_cfg)

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

        logger.info("TML training complete; generating predictions, metrics and plots")
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
        save_options(path=gnn_training_opts_path, options=gnn_cfg)

        logger.info("Loading graph dataset for GNN training")
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
            # Polymer descriptors use the pipeline-wide feature scaler.
            preprocessing_cfg=preprocessing_cfg,
        )

        logger.info("GNN training complete; generating predictions, metrics and plots")
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
        # Same merge as the CLI: the set label comes from the GNN predictions.
        # TML reports the validation samples as Training when they were used
        # for TML training, so merging on the set would drop them.
        predictions = pd.merge(
            left=gnn_predictions_df.drop(columns=[label_col_name]),
            right=tml_predictions_df.drop(columns=[ResultColumn.SET]),
            on=[ResultColumn.INDEX, iterator],
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

    logger.info("Saving predictions and metrics")
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

    if train_gnn_options.exists() or train_tml_model_options_path(experiment_path).exists():
        display_model_results(experiment_path=experiment_path, expanded=False)

    advanced = st.toggle(
        "Unlock advanced training options",
        value=False,
        key=f"advanced_training:{experiment_name}",
        help="Unlock hyperparameters, tuning grids, splitting, preprocessing and optimisation. "
        "Locking again restores the GUI defaults for the next run.",
    )
    if not advanced:
        st.info(
            "Advanced options are locked. GUI defaults are used; "
            "unlock above to customise them. These defaults are a starting point, "
            "not a guarantee of the best model for every dataset."
        )

    st.markdown("## Train Machine Learning Models (TMLs)")

    if representation_file_path(experiment_path=experiment_path).exists():

        tml_models = train_TML_models(problem_type=data_opts.problem_type, advanced=advanced)

    else:
        st.error("No descriptors representation found, TML models cannot be trained.")
        tml_models = {}

    st.markdown("## Graph Neural Networks (GNNs)")

    if gnn_raw_data_path(experiment_path=experiment_path).exists():

        gnn_conv_params = train_GNN_models_form(
            representation_opts=representation_opts,
            problem_type=data_opts.problem_type,
            advanced=advanced,
        )
        gnn_optimisation_cfg = (
            (
                gnn_optimisation_widgets(problem_type=data_opts.problem_type)
                if advanced
                else GNNOptimisationConfig()
            )
            if gnn_conv_params
            else None
        )

    else:
        st.error(
            "No graph representation found. Please build a graph representation of your polymers first."
        )
        gnn_conv_params = {}
        gnn_optimisation_cfg = None

    # ------------------------------------------------------------------
    # Feature preprocessing (pipeline-wide): only meaningful when there are
    # tabular features to scale — TML descriptors and/or GNN polymer descriptors.
    # ------------------------------------------------------------------
    gnn_polymer_descriptors = bool(gnn_conv_params) and bool(
        representation_opts.polymer_descriptors
    )
    preprocessing_cfg = None
    if tml_models or gnn_polymer_descriptors:
        st.markdown("## Feature Preprocessing")
        preprocessing_cfg = (
            feature_transformer_widgets(
                train_tml=bool(tml_models), gnn_polymer_descriptors=gnn_polymer_descriptors
            )
            if advanced
            else default_feature_preprocessing(train_tml=bool(tml_models))
        )
        if not advanced:
            st.caption(
                "MinMax scaling; "
                + (
                    "variance filter (threshold 0.0), then correlation filter (threshold 0.95)."
                    if tml_models
                    else "no feature selection for GNN polymer descriptors."
                )
            )

    st.markdown("## Data Splitting Options")
    valid_split = split_data_form(problem_type=data_opts.problem_type, advanced=advanced)

    # ------------------------------------------------------------------
    # Target variable scaling (regression only)
    # ------------------------------------------------------------------
    target_cfg = TargetTransformConfig()
    if data_opts.problem_type.lower() == "regression":
        st.markdown("## Target Variable Scaling")
        if advanced:
            target_cfg = target_transform_widget()
        else:
            st.caption("No target scaling (existing GUI default).")

    if (gnn_conv_params or tml_models) and valid_split:
        disabled = False
    else:
        disabled = True

    if tml_models or gnn_conv_params:
        raw_path = gnn_raw_data_file(file_name=data_opts.data_name, experiment_path=experiment_path)
        source_data = read_csv(raw_path if raw_path.exists() else data_opts.data_path, index_col=0)
        workload = estimate_workload(
            rows=len(source_data),
            representations=len(representation_opts.molecular_descriptors),
            tml_models=len(tml_models),
            gnn_models=len(gnn_conv_params),
            repetitions=st.session_state.get(GeneralConfigStateKeys.BootstrapIterations, 1),
            tml_samples=(
                st.session_state.get(TrainTMLStateKeys.HPONumSamples, DEFAULT_HPO_NUM_SAMPLES)
                if st.session_state.get(TrainTMLStateKeys.PerformHyperparameterTuning)
                else 0
            ),
            gnn_samples=(
                st.session_state.get(TrainGNNStateKeys.HPONumSamples, DEFAULT_HPO_NUM_SAMPLES)
                if st.session_state.get(TrainGNNStateKeys.HypTunning)
                else 0
            ),
            tml_folds=st.session_state.get(TrainTMLStateKeys.HPONumFolds, 5),
            epochs=st.session_state.get(TrainGNNStateKeys.Epochs, 250),
        )
        st.caption(
            f"Workload: {len(source_data):,} rows; {workload.final_fits:,} final model fits; "
            f"up to {workload.fit_budget:,} fits including tuning/folds; "
            f"up to {workload.gnn_epochs:,} GNN epochs. "
            "A workload budget, not a runtime estimate; grid limits, cached searches "
            "and early stopping can reduce it."
        )
        if workload.reasons:
            st.warning(
                "This run may take a long time: "
                + "; ".join(workload.reasons)
                + ". Consider fewer models, splits or tuning trials, or run the calculation "
                "on an HPC system. Use a CUDA-enabled GPU for GNN training; "
                "traditional ML training here primarily uses CPUs. Training remains available."
            )

    confirmed = rerun_warning(
        stage_rerun_plan(experiment_path, "training"), f"training:{experiment_path}"
    )
    if st.button("Run Training", disabled=disabled or not confirmed):
        st.write("Training models...")

        train_models(
            experiment_name=experiment_name,
            tml_models=tml_models,
            preprocessing_cfg=preprocessing_cfg,
            gnn_conv_params=gnn_conv_params,
            gnn_optimisation_cfg=gnn_optimisation_cfg,
            representation_options=representation_opts,
            data_options=data_opts,
            target_cfg=target_cfg,
        )

        st.success("Models trained successfully!")
