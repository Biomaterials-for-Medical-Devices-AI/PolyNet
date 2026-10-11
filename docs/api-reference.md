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
├── inference/       # Prediction assembly, unseen-data prediction (predict_unseen_*), split ensembles (ensemble.py)
├── applicability/   # Applicability domain of new polymers: kNN domain (Ruzicka or Euclidean), expected error from test errors
├── pipeline/        # Shared pipeline stage functions (stages.py) the YAML runner behind `polynet run` (runner.py) and `polynet check` (self_check.py)
├── explainability/  # Attribution
│   ├── masking.py          # GNN chemistry-masking attribution
│   ├── explain.py          # compute_global_attribution(), compute_local_attribution() (GNN)
│   ├── shap_explain.py     # compute_global_shap_attribution(), compute_local_shap_attribution() (TML)
│   ├── embeddings.py       # Graph embedding extraction (PCA, t-SNE) — not currently exposed in the GUI
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

### GUI support modules (`polynet.app`)

These helpers belong to the optional Streamlit layer, not the core pipeline:

| Module | Public entry points | Responsibility |
|---|---|---|
| `services.cache` | `file_version`, `read_csv`, `read_json` | Bounded copy-on-read caches keyed by resolved path, size and nanosecond modification/change timestamps. Missing files raise; rewritten files refresh. CSV keyword arguments participate in the key. |
| `services.reruns` | `RerunPlan`, `stage_rerun_plan`, `rerun_warning`, `clear_stage_results`, `explanation_rerun_plan`, `explanation_rerun_button`, `clear_explanation_results` | GUI-only warnings and acknowledgement before replacing results. Representation/training invalidate downstream generated outputs; explanation replacement preserves unrelated analyses, classes, model/sample pairs and data sources. Cache rewrites are atomic; inspection never deletes files. |
| `services.progress` | `run_progress`, `RunLogHandler` | Context manager/decorator for session-thread logging, an 80-message tail, throttled epoch rendering, and handler/level cleanup on success or failure. |
| `services.explanation_results` | `capture_analysis`, `list_saved_analyses`, `load_analysis`, `analysis_summary`, `analysis_label`, `attribution_coverage`, `attribution_set_membership`, `select_attribution_coverage`, `local_prediction_breakdowns` | Atomic JSON snapshots of completed explanations; population and molecule views share saved scores. Cached selection summaries omit dates; coverage, per-bootstrap set membership and prediction filters identify the actual selected model/sample pairs. No fitted-model loading or inference. |
| `components.forms.explanation_plots` | `explanation_plots_form` | Page 6 saved-analysis selection and all GNN/TML plot controls; both views select architectures/models, bootstraps, sets and exact combinations. Renders saved scores and matching prediction panels. |
| `components.forms.fragment_plots` | `saved_fragment_table`, `fragment_exports` | Page 6 SME F tags fixed across the entire saved analysis; CSV mapping and a separate configurable RDKit SVG/PNG structure key. Bounded drawing caches avoid repeated exports on unrelated widget changes. |
| `services.workload` | `estimate_workload`, `Workload` | Pure calculation of final fits, tuning-inclusive fit/epoch budgets and advisory warning reasons. No training, timing prediction or device allocation. |
| `components.forms.representation` | `default_molecular_descriptors`, `basic_representation`, `cached_graph_feature_analysis` | Input-type-aware GUI descriptor defaults, locked representation controls and cached legacy graph analysis. |
| `components.forms.train_models` | `default_feature_preprocessing` | GUI MinMax/variance/correlation baseline; GNN descriptors receive scaling only. |
| `components.plots` | `display_model_results(..., external=False)` | On-demand training results; pass `external=True` for external dataset folders, including legacy layouts. |

`services.configurations.load_options` validates freshly copied cached JSON;
`save_options` still delegates to core configuration I/O. Core training remains
Streamlit-free. `training.gnn.train_model(log_context=...)` adds optional context
to epoch messages, and `training.hyperopt.TrainingProgressCallback` forwards Ray
trial activity through normal logging from the driver.

The focused regression coverage is in `tests/test_gui_feedback.py`: cache
freshness/copy isolation, session log isolation/cleanup, workload boundaries,
locked/unlocked defaults, external result layouts, parity legend styling and
GUI training reproducibility.

The core explainability package also exports `render_cached_global_attribution`,
`render_cached_local_attribution`, `render_cached_global_shap_attribution` and
`render_cached_local_shap_attribution`. These accept raw attribution dictionaries/frames,
model names and plotting inputs, and return the same result dataclasses as the existing
`compute_*` functions. They have no Streamlit dependency, load no fitted models, never
perform inference, and do not update caches. The existing computation APIs retain their
signatures and delegate rendering to these helpers.
Local GNN rendering accepts optional `mols_per_model`; local SHAP rendering accepts
optional `samples_per_model`. These restrict displayed pairs to each model/bootstrap's
chosen samples while retaining the full saved population for PerModel normalisation.

SME `render_cached_global_attribution` and the ridge/bar/strip plotting functions accept
optional `fragment_labels={original_identifier: display_label}`. Display labels never
replace raw keys or affect numerical values and filtering.
The renderer also accepts `fragment_label_color` (a Matplotlib colour string) and
`match_fragment_label_colors` (matching takes precedence). Its returned
`GlobalAttributionResult.fragment_label_colors` maps original identifiers to RGB
colours for consistent structure exports. The three plotting functions accept this
mapping as `fragment_label_colors`. Leaving these arguments unset retains their
existing defaults. `visualization.attribution_fragment_colors` computes the shared
ridge/bar palette or strip mean-marker colours for the selected top/bottom fragments.

`polynet.explainability.fragment_figures` (also exported from `polynet.explainability`)
provides `fragment_label_table`, `FragmentGridOptions`, `fragment_structure_svg` and
`fragment_structure_png`. Supply all saved fragment identifiers to `fragment_label_table`
for stable tags. Structure renderers take a selected mapping DataFrame in plot order.
`FragmentGridOptions` defines columns, panel width/height and gaps in mm, atom/tag fonts
and bond width in pt, and bond length in mm. PNG defaults to 600 DPI and embeds resolution
metadata; SVG is vector. `ordered_attribution_fragments` returns the shared descending
mean/top-bottom selection used by the distribution and its exports.
Both structure renderers accept optional `tag_colors={original_identifier: colour}`,
defaulting missing tags to black. These colours affect tag text only. GUI drawing
caches include this mapping so changing colours refreshes SVG and PNG exports.
