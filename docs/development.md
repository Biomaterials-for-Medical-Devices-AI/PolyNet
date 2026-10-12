# Development

- [Testing](#testing)
- [Running the GUI while developing](#running-the-gui-while-developing)
- [Extending PolyNet](#extending-polynet)

---

## Testing

PolyNet ships with a [pytest](https://docs.pytest.org/) suite under `tests/`. Most tests
need only RDKit (no GPU, PolyBERT model, or PolyMetriX install) and run quickly.

### Running the tests

```bash
# Activate your environment first, then:
python -m pytest tests/ -v

# A single module
python -m pytest tests/test_merging.py -v

# List collected tests without running them
python -m pytest tests/ --co -q
```

### Test modules

| Module | What it covers |
|---|---|
| `test_fragment_figures.py` | Stable SME F tags and generated SMARTS, ridge/bar/strip labels and observation-count positions through top/bottom selection without score changes, custom and distribution-matched tag colours without recolouring data or molecules, vector SVG and physical layout, PNG resolution metadata, invalid/oversized exports, and page 6 CSV/structure exports without inference or cache writes |
| `test_explanation_exports.py` | Every SME/SHAP/results plot style exported as SVG/PDF/PNG, embedded fonts and vector artist preservation, restoration of rasterisation flags, PNG DPI metadata/resolution, native RDKit heatmap redraws, per-descriptor/molecule/model export buttons, independent simultaneous prediction/metric matrix controls, original training PNG bytes and invalid/oversized export settings |
| `test_explanation_results.py` | Saved GNN/TML replay without inference, exact class/split populations, external roots, numeric IDs, native plot styles, population/molecule views, architecture/bootstrap/exact-combination selection, each bootstrap's own set membership in plots and prediction panels, additive SHAP baselines, stable PerModel scaling after set filtering, unique population counts and page 5/6 Streamlit interactions |
| `test_gui_reruns.py` | GUI rerun warnings/acknowledgement, dependency cleanup, source/config preservation, selected explanation replacement, classes/fragmentations/bootstrap populations, external isolation, floating-point preservation, numeric-ID cache merges preserving other GNN scores, and invalid splits leaving old results intact |
| `test_gui_feedback.py` | GUI defaults by string type, preprocessing defaults, cache freshness, session log isolation, workload warnings, external results layouts, parity legend styling and training PNG exports, and repeat-training predictions |
| `test_merging.py` | Descriptor merging strategies (`merge_weighted`, `_merge_concatenate`, `_single_smiles`, `_merge` dispatch) |
| `test_compute_rdkit.py` | RDKit descriptor computation across all merging strategies |
| `test_descriptors_regression.py` | Numerical regression against reference CSVs in `tests/fixtures/` |
| `test_config_loader.py` | Config normalisation / migration in `build_experiment_config` |
| `test_persistence.py` | `load_dataframes` column validation (I/O mocked) |
| `test_chem_utils.py` | Chemistry utilities (SMILES handling, fragmentation matching) |
| `test_attachment_point_feature.py` | PSMILES `IsAttachmentPoint` wildcard stripping |
| `test_per_monomer_pool.py` | Per-monomer pooling weighting |
| `test_explain_source.py` | Explainability source/experiment resolution |
| `test_shap_class_selection.py` | SHAP per-class value selection for classification |
| `test_pipeline_regression.py` | End-to-end pipeline regression (GNN + TML) against fixtures |
| `test_cli.py` | Streamlit options passed by `polynet` / `polynet gui` (file watcher off unless set) |

Shared fixtures live in `tests/conftest.py` (e.g. `two_monomer_df`, `single_monomer_df`,
`known_df_dict`). Reference CSVs for the descriptor regression tests live in
`tests/fixtures/`; pipeline regression fixtures live in `tests/fixtures/pipeline/`.

### Writing new tests

- Place test files in `tests/` with the `test_` prefix; add shared fixtures to
  `tests/conftest.py`.
- Prefer testing the public API (`compute_rdkit_descriptors`,
  `build_vector_representation`, the `compute_*_attribution` functions) over private
  helpers unless the private logic is complex enough to warrant direct testing.
- Tests that require PolyBERT or a GPU should be marked and skipped in CI by default.

## Running the GUI while developing

`polynet` starts the GUI with Streamlit's file watcher **off**: users never edit the app's
source, and the watcher's module scan floods the terminal with harmless tracebacks with
recent `transformers` versions (below). When working on PolyNet, turn it back on so the app
reloads when you save a file:

```bash
polynet --server.fileWatcherType auto
```

`streamlit run polynet/app/Welcome_to_PolyNet.py` from the repository root does the same.

With the watcher on, Streamlit inspects every loaded module on each rerun and logs any
module that fails to import as `Examining the path of <module> raised:` with a traceback.
Read the last line of the traceback:

- `No module named 'torchvision'` (or another optional package) raised from a
  `transformers.models.…` module is expected and harmless: `transformers` (loaded for
  polyBERT) has modules for many models PolyNet does not use.
- A `polynet.…` module, or a package PolyNet imports directly, failing to import is a real
  bug: a missing dependency or a broken import that the page you are testing may not have
  reached yet.

The development environment uses the versions pinned in `poetry.lock`, while
`pip install polynet` takes the newest versions allowed by `pyproject.toml`. Before a
release, check the GUI and `polynet check` in a fresh environment installed from the built
wheel (`python -m build`, then `pip install dist/polynet-<version>-py3-none-any.whl`), with
the watcher on, to catch problems that only appear with newer dependencies.

## Extending PolyNet

### Adding a new GNN architecture

1. Create `polynet/models/gnn/myarch.py` following the pattern in `gcn.py`.
2. Add `MyArchBase`, `MyArchClassifier`, `MyArchRegressor`.
3. Register in `polynet/models/gnn/__init__.py`.
4. Add `(Network.MyArch, ProblemType.Regression)` and
   `(Network.MyArch, ProblemType.Classification)` to `_NETWORK_REGISTRY` in
   `polynet/factories/network.py`.
5. Add `MyArch` to the `Network` enum in `polynet/config/enums.py` (and, optionally, a
   display label to `polynet/config/display_names.py`).

### Adding a new TML model

1. Add the model class to `_TML_REGISTRY` in `polynet/training/tml.py`.
2. Add the identifier to the `TraditionalMLModel` enum in `polynet/config/enums.py`
   (and, optionally, a display label to `polynet/config/display_names.py`).

### Adding a new attribution algorithm

1. Add the identifier to the `ExplainAlgorithm` enum in `polynet/config/enums.py`.
2. Implement the computation in `polynet/explainability/` — returning results as plain
   Python data structures (no Streamlit dependency).
3. Wire it up in `polynet/explainability/explain.py` and export from
   `polynet/explainability/__init__.py`.
4. Update the `ExplainabilityConfig` validator in
   `polynet/config/schemas/explainability.py` to accept the new value.
