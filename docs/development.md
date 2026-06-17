# Development

- [Testing](#testing)
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
