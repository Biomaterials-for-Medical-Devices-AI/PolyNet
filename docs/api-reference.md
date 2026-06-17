# API Reference

- [Project structure](#project-structure)
- [Package API](#package-api)

---

## Project structure

A directory-level map of the core package. (Function names within modules evolve — treat
the inline notes as a guide, not an exhaustive index.)

```
polynet/
├── config/          # Pydantic schemas, enums, constants, IO, and path helpers
│   ├── enums.py            # ProblemType, Network, MolecularDescriptor, ImportanceNormalisationMethod,
│   │                       # ShapGlobalPlotType, ExplanationAggregation, HpoSplitStrategy, …
│   ├── constants.py        # ResultColumn and shared string constants
│   ├── column_names.py     # Standardised prediction/score column-name builders
│   ├── display_names.py    # Human-readable model/descriptor/metric labels (+ abbreviations)
│   ├── io.py               # save_options(), load_options()
│   ├── paths.py            # Path-helper functions for experiment directories
│   ├── from_yaml.py        # YAML → ExperimentConfig loader
│   ├── _loader.py          # Internal config merging/normalisation
│   ├── search_grid.py      # Default HPO search grids
│   └── schemas/            # Per-section Pydantic models (data, representation, training,
│                           # split_data, target/feature preprocessing, explainability,
│                           # tml_explainability, …)
│
├── data/            # Data loading and preprocessing
│   ├── loader.py           # load_dataset() — CSV loading with validation
│   ├── preprocessing.py    # sanitise_df(), TargetScaler
│   └── feature_transformer.py  # FeatureTransformer (scaler + selection; NaN/inf-safe)
│
├── experiment/      # Experiment filesystem management (get_experiments, create_experiment)
├── featurizer/      # Molecular feature generation
│   ├── descriptors.py      # build_vector_representation() — RDKit (PSMILES-capped), PolyBERT, Morgan, …
│   ├── polymer_graph.py    # CustomPolymerGraph (PyG graph builder)
│   └── pmx.py              # PolyMetriX integration
│
├── models/          # Network classes (gnn/) and model/scaler persistence
├── factories/       # Object construction (network, dataloader, loss, optimizer)
├── training/        # Training loops
│   ├── gnn.py              # train_gnn_ensemble()
│   ├── tml.py              # train_tml_ensemble()
│   ├── metrics.py          # evaluation metrics
│   └── hyperopt.py         # gnn_hyp_opt() — Ray Tune + ASHA HPO for GNN
│
├── inference/       # Prediction assembly + unseen-data prediction (predict_unseen_*)
├── pipeline/        # Shared pipeline stage functions (stages.py)
├── explainability/  # Attribution
│   ├── masking.py          # GNN chemistry-masking attribution
│   ├── explain.py          # compute_global_attribution(), compute_local_attribution() (GNN)
│   ├── shap_explain.py     # compute_global_shap_attribution(), compute_local_shap_attribution() (TML)
│   ├── embeddings.py       # Graph embedding extraction (PCA, t-SNE)
│   └── visualization.py    # Shared plotting helpers
│
├── plotting/        # Data-exploration plots
├── visualization/   # Low-level rendering (parity, ROC, confusion, p-value matrix, box plots)
├── utils/           # General utilities
│   └── statistical_analysis.py  # pairwise p-value matrices, correct_pvalue_matrix(),
│                                # MULTIPLE_COMPARISON_METHODS, significance_marker()
│
└── app/             # Streamlit GUI (optional; requires streamlit)
    ├── Welcome_to_PolyNet.py
    ├── pages/                # 1_Create_Experiment … 6_Analyse_Results
    ├── components/forms/     # explain_model.py (GNN), explain_tml.py (TML), analyse_results.py, …
    ├── services/             # Thin re-exports of core modules + Streamlit rendering for explanations
    └── options/              # Re-exports + Streamlit session-state keys

scripts/
├── run_pipeline.py        # CLI entry point
└── integration_test.py    # Staged integration test for debugging

configs/experiment.yaml    # Template experiment configuration
tests/                     # pytest suite (see development.md)
```

## Package API

### `polynet.pipeline`

The main entry point for running experiment stages. Both the CLI and the Streamlit app
call these functions.

```python
from polynet.pipeline import (
    build_graph_dataset,    # Featurise data into a PyG graph dataset
    compute_descriptors,    # Compute molecular descriptor vectors and save CSVs
    compute_data_splits,    # Generate bootstrap train/val/test split indices
    train_gnn,              # Train a GNN ensemble → (trained, loaders, target_scalers)
    run_gnn_inference,      # Run GNN inference on all splits → predictions DataFrame
    train_tml,              # Train a TML ensemble → (trained, data, scalers, target_scalers)
    run_tml_inference,      # Run TML inference on all splits → predictions DataFrame
    compute_metrics,        # Compute evaluation metrics from predictions DataFrame
    plot_results_stage,     # Generate learning curves and result plots
    predict_external,       # Predict on unseen data using a trained experiment
    run_explainability,     # Run GNN chemistry-masking attribution and save heatmaps
    run_tml_explainability, # Run TML SHAP attribution and save plots
)
from polynet.config.schemas import TargetTransformConfig   # Target scaling config (regression)
from polynet.config.enums import TargetTransformDescriptor # Enum of scaling strategies
```

Key signatures for the training and inference stages:

```python
# train_gnn returns a 3-tuple; pass target_cfg to enable target scaling
gnn_trained, gnn_loaders, gnn_target_scalers = train_gnn(
    dataset, split_indexes, data_cfg, gnn_cfg,
    random_seed=42, out_dir=experiment_path,
    target_cfg=TargetTransformConfig(strategy=TargetTransformDescriptor.StandardScaler),
)

# run_gnn_inference — pass target_scalers to inverse-transform predictions
gnn_preds_df = run_gnn_inference(
    trained_models=gnn_trained, loaders=gnn_loaders,
    data_cfg=data_cfg, split_cfg=split_cfg,
    target_scalers=gnn_target_scalers,
)

# train_tml returns a 4-tuple
tml_trained, tml_training_data, tml_scalers, tml_target_scalers = train_tml(
    desc_dfs=dataframes, split_indexes=split_indexes,
    data_cfg=data_cfg, tml_cfg=tml_cfg, preprocessing_cfg=preprocessing_cfg,
    random_seed=42, out_dir=experiment_path,
    target_cfg=TargetTransformConfig(strategy=TargetTransformDescriptor.Log10),
)

# run_tml_inference — pass target_scalers to inverse-transform predictions
tml_preds_df = run_tml_inference(
    trained=tml_trained, training_data=tml_training_data,
    data_cfg=data_cfg, split_cfg=split_cfg,
    target_scalers=tml_target_scalers,
)
```

### `polynet.config.io`

```python
from polynet.config.io import save_options, load_options

save_options(path, options)          # Save dataclass, Pydantic model, or dict → JSON
load_options(path, options_class)    # Load JSON → Pydantic model instance
```

### `polynet.config.paths`

Path helpers for all experiment directories and files. Every path used by the pipeline
is constructed here.

```python
from polynet.config.paths import (
    polynet_experiments_base_dir,           # ~/PolyNetExperiments
    polynet_experiment_path,                # ~/PolyNetExperiments/<name>
    data_options_path,                      # <experiment>/data_options.json
    representation_options_path,            # <experiment>/representation_options.json
    train_gnn_model_options_path,           # <experiment>/train_gnn_options.json
    train_tml_model_options_path,           # <experiment>/train_tml_options.json
    model_dir,                              # <experiment>/ml_results/models/
    plots_directory,                        # <experiment>/ml_results/plots/
    ml_results_file_path,                   # <experiment>/ml_results/predictions.csv
    model_metrics_file_path,                # <experiment>/ml_results/metrics.json
    explanation_parent_directory,           # <experiment>/explanations/
    unseen_predictions_experiment_parent_path,  # <experiment>/unseen_predictions/<file>/
    # ... and more helpers
)
```

### `polynet.experiment.manager`

```python
from polynet.experiment.manager import get_experiments, create_experiment

experiments = get_experiments()                       # List all saved experiments
create_experiment(experiment_path, data_options)      # Create dir and save data config
```

### `polynet.models.persistence`

```python
from polynet.models.persistence import (
    save_gnn_model, load_gnn_model,                  # .pt files via torch.save/load
    save_tml_model, load_tml_model,                  # .joblib files via joblib
    load_models_from_experiment,                      # Load all .pt and .joblib from models/
    load_scalers_from_experiment,                     # Load all .pkl scalers from models/
    load_dataframes,                                  # Load and validate descriptor CSVs
)
```

### `polynet.inference.predict`

```python
from polynet.inference.predict import predict_unseen_tml, predict_unseen_gnn

# TML: run all saved TML models on new descriptor DataFrames
preds_df = predict_unseen_tml(models, scalers, descriptor_dfs, data_cfg)

# GNN: run all saved GNN models on a new PolymerGraphDataset
preds_df = predict_unseen_gnn(models, dataset, data_cfg)
```

### `polynet.utils`

```python
from polynet.utils import (
    create_directory,        # mkdir -p, no-op if already exists
    save_data,               # Save DataFrame to .csv or .xlsx
    filter_dataset_by_ids,   # Subset a PyG dataset to a list of sample IDs
    extract_number,          # Parse trailing integer from a model filename (e.g. "model_3.pt")
)
from polynet.utils.statistical_analysis import (
    significance_marker,        # p-value → "*", "**", "***"
    correct_pvalue_matrix,      # Apply a multiple-comparison correction to a p-value matrix
    MULTIPLE_COMPARISON_METHODS,  # Label → statsmodels method name
)
from polynet.config.display_names import prettify_label, prettify_metric
```
