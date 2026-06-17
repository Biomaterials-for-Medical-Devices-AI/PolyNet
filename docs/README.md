# PolyNet Documentation

Detailed documentation for [PolyNet](../README.md). The top-level README covers the
overview, installation, and a quick start; everything else lives here.

| Document | Contents |
|---|---|
| [Configuration reference](configuration.md) | Every section of the YAML config: `experiment`, `data`, `representations`, `splitting`, `gnn_training` (+ automatic HPO), `tml_models`, `target_transform`, `explainability`, `tml_explainability`, `prediction` |
| [Descriptors](descriptors.md) | Polymer-level descriptor fusion and the PolyMetriX integration (side chain / backbone / full repeat unit) |
| [Explainability](explainability.md) | GNN chemistry-masking attribution and TML SHAP attribution — normalisation, per-model vs averaged display, native SHAP plots, prediction breakdowns, caching, and the statistical model comparison on the Analyse Results page |
| [Running the pipeline & outputs](running-and-outputs.md) | CLI flags, predicting on external data, target-variable scaling, the output directory layout, and the debugging integration test |
| [API reference](api-reference.md) | Project structure and the public `polynet` Python API |
| [GUI & architecture](gui-and-architecture.md) | The Streamlit application pages and the core-vs-GUI separation |
| [Development](development.md) | Running the test suite and extending PolyNet (new architectures, TML models, attribution algorithms) |
