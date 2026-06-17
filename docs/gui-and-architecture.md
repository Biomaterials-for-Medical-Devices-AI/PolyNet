# Streamlit GUI & Architecture

- [Streamlit GUI](#streamlit-gui)
- [Architecture: core vs GUI](#architecture-core-vs-gui)

---

## Streamlit GUI

Launch with:

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

Both the GNN and TML sections share architecture / bootstrap / representation selectors
consistent with the training UI, and offer the same options:

- **Global tab** — GNN fragment-attribution distribution (ridge / bar / strip); TML
  native SHAP summary (beeswarm / bar / violin).
- **Local tab** — per-molecule / per-sample explanations with a *Models display*
  toggle: **Average across models** (one plot per molecule) or **Show each model
  separately** (one plot per model × molecule). Each panel shows the molecule's true
  label and a per-(model × bootstrap) table of predicted value and set membership. TML
  local plots are native SHAP waterfall / force / bar, honour the chosen positive /
  negative colours, and have a "top N features" selector.

See [Explainability](explainability.md) for the underlying behaviour and the
normalisation options.

### Analyse Results (page 6)

Works for TML-only, GNN-only, and hybrid experiments — model lists are derived from the
predictions table. Beyond parity plots and confusion matrices, the page offers pairwise
statistical comparison of models (predictions and metrics) with a selectable
multiple-comparison correction (Holm-Bonferroni by default). Model and metric names are
shown with clean display labels (with an optional abbreviated mode). See
[Statistical model comparison](explainability.md#statistical-model-comparison-analyse-results).

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
