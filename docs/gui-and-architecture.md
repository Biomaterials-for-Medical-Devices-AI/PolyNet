# Streamlit GUI & Architecture

- [Streamlit GUI](#streamlit-gui)
- [Architecture: core vs GUI](#architecture-core-vs-gui)

---

## Streamlit GUI

Launch with:

```bash
polynet
```

This works from any directory once the package is installed. Any `streamlit run` option is forwarded, so `polynet --server.port 8502` changes the port. The equivalent explicit command still works:

```bash
streamlit run polynet/app/Welcome_to_PolyNet.py
```

The app is organised into six sequential pages:

| Page | Purpose |
|---|---|
| **1 — Create Experiment** | Name the experiment, upload data, configure target and SMILES columns |
| **2 — Representation** | Select molecular descriptors (RDKit, Morgan, PolyBERT, PolyMetriX) and GNN featurisation options |
| **3 — Train Models** | Configure GNN architectures and TML models, set data splits, optionally apply target-variable scaling (regression), start training |
| **4 — Predict** | Upload an unseen CSV and run prediction using a trained experiment |
| **5 — Explain Models** | Run GNN fragment-masking attribution and TML SHAP attribution, each with global and local tabs (see below) |
| **6 — Analyse Results** | Inspect predictions and metrics, parity plots / confusion matrices, and statistical model comparisons |

The GUI is an optional extension of the core package. `polynet.app` can be omitted
entirely — the core pipeline works without Streamlit.

### Explain Models (page 5)

Choose models, bootstrap repetitions, representations, target class, fragmentation
(GNN), and samples here. The **Global** and **Local** tabs calculate raw importances,
save a completed-analysis snapshot, and show default previews. Defaults are PerModel
normalisation, cyan/red colours, ridge/top-and-bottom 10 for GNN global,
beeswarm/top 10 for TML global, averaged local explanations, and waterfall/top 15
for TML local. Calculations run only when their button is pressed; the automatic-running
toggles have been removed. Plot customisation belongs to page 6.

### Analyse Results (page 6)

Works for TML-only, GNN-only, and hybrid experiments. Alongside parity plots,
confusion matrices and statistical comparisons, **Feature importance plots** detects
completed explanations for the experiment and external datasets. Enable
**Customise feature importance plots** and select a completed analysis, which loads
automatically. A single available analysis requires no extra selector. Analysis labels
show selections and models, without dates. Choose **Population** or **Individual
molecules** to view the same saved data.

All existing presentation controls are available here: Local/PerModel/Global/no
normalisation; positive/negative colours; GNN ridge/bar/strip and top/bottom N;
TML beeswarm/bar/violin and top N; local Average/Separate model display; TML
waterfall/force/bar and feature limit; and custom GNN monomer names. Predictions,
per-bootstrap set membership and attribution tables remain in the local panels.
Changing a control redraws saved scores without loading models or running inference.

Snapshots retain the exact selected samples and each model's own split population.
For individual molecules, choose which cached model/bootstrap combinations to show and
whether to average them as an ensemble or show them separately. Each molecule clearly
lists the available combinations used in its plot; missing pairs are excluded. PerModel
uses the model's maximum absolute score across the whole saved analysis, before averaging.
The older GUI/CLI raw-cache checkbox has been removed. Run page 5 again if saved SHAP
results predate baseline storage or if additional scores are required.

Parity legends explicitly use an opaque white frame, normal text, a grey parity line,
and no shadow, including PNGs saved automatically during training. Previously saved PNGs
retain their old appearance; regenerate plots from saved predictions without retraining.
Page 6 preserves the rendered parity figure so saving it does not produce a blank image.

See [Explainability](explainability.md) for attribution and normalisation behaviour,
and [Statistical model comparison](explainability.md#statistical-model-comparison-analyse-results)
for the model comparison options.

## Architecture: core vs GUI

All reusable logic lives in the core `polynet` package:

```
polynet/                  ← no Streamlit dependency
├── pipeline/stages.py    ← all pipeline stages
├── config/io.py          ← config serialisation
├── config/paths.py       ← path helpers
├── experiment/manager.py ← experiment filesystem
├── models/persistence.py ← model save/load
├── inference/predict.py  ← prediction on new data
└── ...

polynet/app/              ← Streamlit (optional)
├── services/             ← thin re-exports of core modules + Streamlit rendering
│   ├── configurations.py → from polynet.config.io import ...
│   ├── model_training.py → from polynet.models.persistence import ...
│   └── ...
└── pages/                ← Streamlit page orchestration
```

Most `polynet.app.services.*` modules are thin re-export shims — they import from core
and re-export under the original names, so existing app pages continue to work
unchanged. Any code that previously imported from `polynet.app.services.configurations`
still works; the implementation now lives in `polynet.config.io`. (The explanation
rendering services — `explain_model.py`, `explain_tml.py` — are Streamlit-coupled and
stay in the app.)

## Default mode, progress and page loading

Representation and Train Models start with advanced options locked. Unlock the
page's advanced-options toggle to change scientific settings. Locking again
restores the GUI defaults for the next build/run; it does not edit a
saved experiment until its action button is pressed. These defaults are a
starting point, not an assertion of optimal settings for every dataset:

- Representation: **SMILES** uses every RDKit descriptor and Morgan count
  fingerprints (2,048 bins, radius 3). **PSMILES** uses polyBERT and all PolyMetriX
  repeat-unit, side-chain and backbone descriptors, with mean side-chain
  aggregation. Polymer-only descriptors are disabled and cleared for SMILES.
  Graphs retain five atom properties, three bond properties, observed allowable
  graph values and no wildcard placeholders.
  Multiple monomers use weighted-average descriptors and weighted graphs; the
  user still supplies their weight columns. Either representation can be disabled.
- Training: existing model/architecture parameters, 250 GNN epochs, no tuning,
  MinMax feature scaling, followed by variance (threshold 0.0) and correlation
  (threshold 0.95) filtering for TML. GNN polymer descriptors use the same scaler
  without selection. Target scaling remains off, with one 60/20/20 split.
  Classification uses stratified assignment; regression uses random assignment.
  Both use the random sampler, without class balancing. TML includes validation
  samples in fitting, as before. Model selection remains available.

Training, representation building, prediction and explanation actions show a
live activity panel with the latest 80 log messages. Training messages identify
the model, descriptor representation where applicable, and split/bootstrap
repetition. GNN epoch messages repeat that context and show current/total epochs.
Ray tuning reports trial starts, results, completions and failures from its
controller. The same core messages appear in CLI logs. A single estimator fit
or cross-validation trial may have no intermediate messages; the activity panel
remains running until the operation finishes or fails.

The training page shows a workload budget before execution. Advisory warnings
appear when **any** of these conservative thresholds is reached:

| Trigger | Threshold |
|---|---:|
| Final fits: models × descriptor representations (TML) × repetitions | 10 |
| Fits including tuning candidates × internal folds plus final refits | 50 |
| Dataset rows, with multiple fits or any GNN | 5,000 |
| GNN epochs summed across final fits and tuning/folds | 2,500 |

These are transparent workload heuristics, not time predictions or memory
limits. Actual tuning may use fewer fits because of grid size, cached searches
or early stopping. The warning suggests reducing the workload or using HPC;
CUDA is recommended for GNN training, while the current traditional ML path
primarily uses CPUs. The warning does not block training.

GUI CSV/JSON reads use bounded data caches keyed by resolved path, file size,
and nanosecond modification/change timestamps. Rewriting or replacing files
(including through the CLI) refreshes the next read. Cached objects are copied,
so edits in one session do not alter another's view. Legacy graph feature
analysis is also cached by input data. Training, inference and fitted models
are not cached. Saved results load only when **Show Model Results** (or the
corresponding external dataset toggle) is enabled, avoiding eager rendering of
all historical plots on each page visit. This improves repeat visits and widget
reruns; first imports and first-time computations still take time.

### Review checklist for these changes

Use a new or copied experiment: the existing training workflow deletes old
results before validating splits/HPO folds. That pre-existing behaviour remains
unchanged in this logging/display update.

1. **Page loading and freshness.** Open Representation, Train Models, Explain
   Models and Analyse Results. Revisit them and change selectors. Previous
   results should remain unloaded until their Show toggle is enabled. After a
   run replaces results, the displayed predictions and metrics should refresh.
2. **Locked defaults.** Check Representation and Train Models with advanced
   options locked. Unlock, inspect/customise options, then lock again. The next
   operation should use the stated defaults. Check classification as well as
   regression; classification retains stratified splitting.
3. **GUI progress.** On a small dataset, unlock training, select two models and
   two bootstrap repetitions, and reduce GNN epochs for a quick check. Confirm
   model/representation/repetition labels, GNN epochs, completion messages and
   saved outputs. Try a small tuning search to see trial activity. Also check
   representation building, prediction and explanation activity panels.
4. **Workload warning.** Without starting an expensive run, select two models
   and five repetitions to reach ten final fits, or enable tuning to exceed
   fifty budgeted fits. Confirm the reason, HPC/CUDA guidance and enabled Run
   Training button. Return to one untuned TML model and one repetition on a small
   dataset; the warning should disappear.
5. **CLI logs and results.** Run a small existing config with multiple models
   and repetitions (`polynet run --config <config.yaml>`), using a fresh output
   directory. Check model and split/bootstrap labels alongside epochs, then
   compare predictions/metrics with the same seed and configuration.
6. **Automated checks.** In the `polynet` conda environment, run
   `python -m pytest tests -q`, `python -m black --check polynet`, and
   `python -m isort --check-only polynet`. The full suite was verified with
   `HF_HUB_OFFLINE=1` using the already-cached PolyBERT model: 433 tests passed,
   including all four numerical pipeline regression tests. Offline mode requires
   that model to be cached; otherwise allow its normal download.

Adjacent GUI bugs fixed while checking these flows: duplicate shared GNN widget
keys; an out-of-range initial value for the Random Forest depth slider; and
TML-only training requiring a graph CSV even when only descriptors were built.
The latter now uses the saved source dataset when no graph CSV exists. The
existing-results notice also covers TML-only experiments.

### External results and parity plots

External dataset toggles read `predictions.csv` and `metrics.json` directly from
`unseen_predictions/<dataset>/`; older `ml_results/` layouts are also supported.
An empty run shows an explicit message. The detailed metric table includes
ensemble rows; mean/std summaries use individual splits only. The parity
reference line is grey and its legend has normal-weight text on an opaque white
background, so inherited plot styling cannot expose points/grid underneath it.
Previously exported images are unchanged; newly rendered plots use this style.

These GUI defaults apply to newly built representations and new training runs.
Existing saved experiments and explicit CLI configurations are not migrated.
Changing descriptor families or preprocessing intentionally changes new results.
