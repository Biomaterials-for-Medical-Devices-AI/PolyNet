# Descriptors

PolyNet builds fixed-length vector representations for traditional ML and graph
representations for GNNs. This document covers two cross-cutting topics:

- [Polymer descriptor fusion](#polymer-descriptor-fusion) — injecting given polymer-level features
- [PolyMetriX descriptors](#polymetrix-descriptors) — polymer-aware chemical descriptors

> **PSMILES / RDKit descriptors:** RDKit descriptors are computed on a *capped*
> molecule — the polymer attachment points (`*`, atomic number 0) are converted to
> hydrogens and folded into the neighbouring atoms' implicit-hydrogen counts before
> the descriptors are calculated. This avoids the massless dummy atoms corrupting
> mass-, valence- and charge-based descriptors (for example, Gasteiger-charge
> descriptors that would otherwise return `NaN`). Molecules without `*` atoms are
> unaffected.

---

## Polymer descriptor fusion

`representations.polymer_descriptors` lets you inject experimental or pre-computed
polymer-level features — measured molecular weight, degree of polymerisation, or any
other numeric column in your CSV — directly into the modelling pipeline alongside the
computed molecular representations.

### How it works

**Vectorial representations (TML)**

After all per-monomer molecular descriptors are computed and merged (weighted average,
concatenation, or no-merging), the selected polymer descriptor columns are
horizontally concatenated to every representation DataFrame. The final feature matrix
seen by the TML model therefore contains both the structurally derived features and
the given polymer-level features.

**Graph representations (GNN)**

Polymer descriptors cannot be part of node or edge features because they characterise
the whole polymer, not individual atoms or bonds. Instead they are stored as a
**graph-level tensor** on each PyG `Data` object during featurisation
(`CustomPolymerGraph`). During the forward pass they are concatenated to the pooled
graph embedding — **after pooling and after any monomer weight multiplication** — so
the FFN readout receives both the learned graph representation and the experimental
polymer context. The first readout layer is automatically widened to accommodate the
extra dimensions.

```
GNN forward pass with polymer descriptors:

  node features → message passing → [optional weighting] → pooling
                                                               ↓
                                              [embedding, dim = embedding_dim]
                                                               ↓
                                     cat([embedding, polymer_descriptors], dim=1)
                                                               ↓
                                              FFN readout → prediction
```

### Configuration

Specify the column names to use in the `representations` section of your config:

```yaml
representations:
  polymer_descriptors:
    - "molecular_weight"
    - "degree_of_polymerisation"
```

The listed columns must be present as numeric columns in your input CSV. They are used
by both the TML pipeline (if `molecular_descriptors` are configured) and the GNN
pipeline (if `node_features` are configured) automatically — no other changes are
required.

Omitting `polymer_descriptors` (or setting it to `null`) disables the feature entirely
with no impact on existing experiments.

### Python API

```python
from polynet.config.schemas import RepresentationConfig

repr_cfg = RepresentationConfig(
    node_features={...},
    edge_features={...},
    polymer_descriptors=["molecular_weight", "degree_of_polymerisation"],
)

# build_graph_dataset picks up the columns automatically
dataset = build_graph_dataset(data=df, data_cfg=data_cfg, repr_cfg=repr_cfg, out_dir=out_dir)
```

---

## PolyMetriX descriptors

PolyNet integrates the [PolyMetriX](https://lamalab-org.github.io/PolyMetriX/) library
to compute polymer-aware chemical descriptors directly from pSMILES strings. Unlike
general-purpose RDKit descriptors, PolyMetriX is aware of the polymer repeat unit
structure and can target three distinct structural regions: the side chains, the
backbone, and the full repeat unit.

> **Requires:** `polymetrix` must be installed in your environment.

### Three computation modes

| Mode | Config key | Wrapper | Aggregation |
|---|---|---|---|
| Side chain | `side_chain` | `SideChainFeaturizer` | Yes — specify via `agg` |
| Backbone | `backbone` | `BackBoneFeaturizer` | No |
| Full repeat unit | `polymer` | `FullPolymerFeaturizer` | No |

All three modes accept the same set of **chemical features**. The `side_chain` mode
additionally accepts topological features (e.g. `num_sidechains`). The `polymer` mode
only supports chemical features — topological features have no whole-polymer
equivalent and will raise a `ValueError`.

### Available chemical features

| Config value | Description |
|---|---|
| `num_hbond_donors` | Number of hydrogen-bond donor groups |
| `num_hbond_acceptors` | Number of hydrogen-bond acceptor groups |
| `num_rotatable_bonds` | Number of rotatable bonds |
| `num_rings` | Total ring count |
| `num_non_aromatic_rings` | Non-aromatic ring count |
| `num_aromatic_rings` | Aromatic ring count |
| `num_atoms` | Atom count |
| `topological_surface_area` | Topological polar surface area (TPSA) |
| `fraction_bicyclic_rings` | Fraction of rings that are bicyclic |
| `num_aliphatic_heterocycles` | Aliphatic heterocycle count |
| `slogpvsa1` | SlogP VSA descriptor 1 |
| `balaban_j_index` | Balaban J connectivity index |
| `molecular_weight` | Molecular weight |
| `sp3_carbon_count` | Count of sp³ carbon atoms |
| `sp2_carbon_count` | Count of sp² carbon atoms |
| `max_estate_index` | Maximum electrotopological state index |
| `smr_vsa5` | SMR VSA descriptor 5 |
| `fp_density_morgan1` | Morgan fingerprint density (radius 1) |
| `halogen_counts` | Halogen atom count |
| `bond_counts` | Total bond count |
| `bridging_rings_count` | Number of bridging rings |
| `max_ring_size` | Size of the largest ring |
| `heteroatom_count` | Count of non-C, non-H atoms |
| `heteroatom_density` | Heteroatom density |

### Available topological features (side chain only)

| Config value | Description |
|---|---|
| `num_sidechains` | Number of side chains |
| `num_backbone` | Number of backbone atoms |
| `sidechain_length_to_star_attachment_distance_ratio` | Sidechain length / star attachment distance |
| `star_to_sidechain_min_distance` | Minimum distance from star atom to side chain |
| `sidechain_diversity` | Diversity score of side-chain structures |

### Available aggregation methods (`agg`)

| Config value | Description |
|---|---|
| `sum` | Sum over all side chains |
| `mean` | Mean over all side chains |
| `max` | Maximum over all side chains |
| `min` | Minimum over all side chains |

### Configuration

The three modes can be used in any combination. Any key may be **omitted entirely** —
only keys that are present and non-empty are computed. The only requirement, enforced
at config-load time, is that at least one of `side_chain`, `backbone` or `polymer`
provides a descriptor (the `agg` key alone does not count). Each of the three
descriptor lists also accepts the sentinel **`"all"`** (as a bare string or a
one-element list, case-insensitive) as a shortcut for "every available descriptor for
this part" — chemical features for `polymer`, chemical + the part-specific topological
features for `side_chain` and `backbone`.

```yaml
representations:
  molecular_descriptors:
    PolyMetriX:
      # Chemical descriptors on the full repeat unit (no aggregation)
      polymer:
        - molecular_weight
        - topological_surface_area
        - num_rotatable_bonds
        - num_hbond_donors
        - num_atoms
        - num_rings
      # Chemical descriptors on side chains, aggregated across all side chains
      side_chain:
        - num_rings
        - molecular_weight
        - num_sidechains          # topological feature — side_chain only
      agg: [sum, mean]

      # Chemical descriptors on the polymer backbone
      backbone:
        - num_atoms
        - topological_surface_area
```

A minimal configuration that uses `"all"` and omits `polymer` entirely:

```yaml
representations:
  molecular_descriptors:
    PolyMetriX:
      side_chain: "all"           # every chemical + sidechain-topological descriptor
      backbone:   "all"           # every chemical + backbone-topological descriptor
      agg: [sum, mean, min, max]
```

### Python API

```python
from polynet.featurizer.pmx import create_pmx_featurizer
from polynet.config.enums import PMXChemFeature, PMXTopoFeature, PMXAggMethod

featurizer = create_pmx_featurizer(
    side_chain_features=[
        PMXChemFeature.NumRings,
        PMXChemFeature.MolecularWeight,
        PMXTopoFeature.NumSideChainFeaturizer,
    ],
    backbone_features=[
        PMXChemFeature.NumAtoms,
        PMXChemFeature.TopologicalSurfaceArea,
    ],
    agg_method=[PMXAggMethod.Sum, PMXAggMethod.Mean],
    polymer_features=[
        PMXChemFeature.MolecularWeight,
        PMXChemFeature.TopologicalSurfaceArea,
        PMXChemFeature.NumRotatableBonds,
    ],
)

# featurize a single polymer
from polymetrix.featurizers.polymer import Polymer
polymer = Polymer.from_psmiles("c1ccccc1[*]CCO[*]")
features = featurizer.featurize(polymer)
labels   = featurizer.feature_labels()
```

### Design notes

- All three modes are combined into a single `MultipleFeaturizer` and featurized in one
  pass per polymer — no redundant SMILES parsing.
- Feature column ordering in the output is: side-chain features → backbone features →
  topological features → polymer (full repeat unit) features.
- Omitting a key (or leaving it as an empty list) is safe — that mode is simply skipped
  with no error.
- The config-schema validator requires that **at least one** of `side_chain`,
  `backbone` or `polymer` actually provides a descriptor (`"all"` or a non-empty list).
  A `PolyMetriX` block that only sets `agg` is rejected at load time.
- The `polymer` mode raises a `ValueError` at featurizer construction time if any
  topological feature is passed, giving a clear message before any computation begins.
