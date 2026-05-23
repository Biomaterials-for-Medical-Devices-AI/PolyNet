"""
polynet.utils.graph_analysis
=============================
Pre-computation and persistence of atom/bond property frequency analysis.

Run once at experiment-creation time so the Representation page can populate
allowable-value defaults immediately — without re-scanning the dataset on
every widget interaction.

JSON structure
--------------
{
  "atom": {
    "<AtomFeature value>": {
      "<smiles_col>": {
        "present_indices": [0, 5, 6, ...],   // indices into the Options list
        "bar_data": [["6", 120], ["8", 45], ...]  // [[str(value), count], ...]
      }
    }
  },
  "bond": { ... same shape ... }
}

``present_indices`` stores which positions in the ``Options`` list for a given
property were actually found in the dataset.  Storing indices rather than raw
values avoids having to serialise RDKit enum objects (HybridizationType,
BondStereo, ChiralTag, …).  When loading, defaults are reconstructed as
``[Options[i] for i in present_indices]``, which returns the correct typed
objects directly from the allowable-sets registry.

``bar_data`` uses ``str(key)`` so every value is JSON-safe; it is only used
for the optional frequency bar charts on the Representation page.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from polynet.config.enums import AtomBondDescriptorDictKey, AtomFeature, BondFeature
from polynet.featurizer.allowable_sets import atom_properties, bond_features
from polynet.utils.chem_utils import count_atom_property_frequency, count_bond_property_frequency


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _encode_col_data(counts: dict, options: Optional[list]) -> dict:
    """Encode one (property, smiles_col) frequency dict into a JSON-safe format.

    Args:
        counts: Raw output from ``count_atom/bond_property_frequency``.
        options: The ``Options`` list from the allowable-sets registry, or
                 ``None`` for boolean/scalar properties.

    Returns:
        Dict with keys ``"bar_data"`` (always) and ``"present_indices"``
        (only when *options* is not ``None``).
    """
    # Bar-chart data: list of [str_label, count] pairs, sorted so numeric keys
    # appear in ascending numeric order in the chart.
    def _sort_key(item):
        try:
            return float(item[0])
        except (ValueError, TypeError):
            return item[0]

    bar_data = sorted(([str(k), v] for k, v in counts.items()), key=_sort_key)

    result: dict = {"bar_data": bar_data}

    if options is not None:
        # Which positions in the Options list are present in the dataset?
        present_set = set(counts.keys())
        present_indices = [i for i, opt in enumerate(options) if opt in present_set]
        result["present_indices"] = present_indices

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_graph_feature_analysis(
    df: pd.DataFrame,
    smiles_cols: List[str],
) -> Dict:
    """Scan *df* and build the full frequency analysis for every atom and bond
    property across every SMILES column.

    Args:
        df: The experiment dataset (already canonicalised if applicable).
        smiles_cols: Names of the SMILES columns in *df*.

    Returns:
        A nested dict ready to be passed to :func:`save_graph_feature_analysis`.
    """
    analysis: Dict = {"atom": {}, "bond": {}}

    for prop, prop_config in atom_properties.items():
        options = (
            prop_config[AtomBondDescriptorDictKey.Options] if prop_config else None
        )
        analysis["atom"][prop] = {}
        for smiles_col in smiles_cols:
            counts = count_atom_property_frequency(df, smiles_col, prop)
            analysis["atom"][prop][smiles_col] = _encode_col_data(counts, options)

    for prop, prop_config in bond_features.items():
        options = (
            prop_config[AtomBondDescriptorDictKey.Options] if prop_config else None
        )
        analysis["bond"][prop] = {}
        for smiles_col in smiles_cols:
            counts = count_bond_property_frequency(df, smiles_col, prop)
            analysis["bond"][prop][smiles_col] = _encode_col_data(counts, options)

    return analysis


def save_graph_feature_analysis(analysis: Dict, path: Path) -> None:
    """Persist *analysis* to *path* as a JSON file.

    Args:
        analysis: Dict produced by :func:`compute_graph_feature_analysis`.
        path: Destination file path (parent directory is created if absent).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(analysis, fh, indent=2)


def load_graph_feature_analysis(path: Path) -> Optional[Dict]:
    """Load a previously saved analysis JSON.

    Args:
        path: Path to ``graph_feature_analysis.json``.

    Returns:
        The parsed dict, or ``None`` if the file does not exist (e.g. the
        experiment was created before this feature was introduced).
    """
    if not path.exists():
        return None
    with open(path, "r") as fh:
        return json.load(fh)


def get_atom_prop_defaults(
    analysis: Dict,
    prop: AtomFeature,
    smiles_col: str,
) -> set:
    """Return the set of *Options* values present in the dataset for an atom property.

    Reconstructs typed values (including RDKit enums) by indexing back into the
    ``atom_properties`` registry — so the returned objects are the same types
    that Streamlit widgets expect.

    Args:
        analysis: Loaded analysis dict.
        prop: The atom property enum value.
        smiles_col: Which SMILES column to look up.

    Returns:
        Set of values from ``atom_properties[prop][Options]`` that were found
        in the dataset, or an empty set if the data is absent.
    """
    prop_config = atom_properties.get(prop)
    if not prop_config:
        return set()
    options = prop_config[AtomBondDescriptorDictKey.Options]
    col_data = analysis.get("atom", {}).get(str(prop), {}).get(smiles_col, {})
    indices = col_data.get("present_indices", [])
    return {options[i] for i in indices if i < len(options)}


def get_bond_prop_defaults(
    analysis: Dict,
    prop: BondFeature,
    smiles_col: str,
) -> set:
    """Return the set of *Options* values present in the dataset for a bond property.

    Args:
        analysis: Loaded analysis dict.
        prop: The bond property enum value.
        smiles_col: Which SMILES column to look up.

    Returns:
        Set of values from ``bond_features[prop][Options]`` that were found
        in the dataset, or an empty set if the data is absent.
    """
    prop_config = bond_features.get(prop)
    if not prop_config:
        return set()
    options = prop_config[AtomBondDescriptorDictKey.Options]
    col_data = analysis.get("bond", {}).get(str(prop), {}).get(smiles_col, {})
    indices = col_data.get("present_indices", [])
    return {options[i] for i in indices if i < len(options)}


def get_bar_data(
    analysis: Dict,
    feature_type: str,
    prop,
    smiles_col: str,
) -> Optional[pd.DataFrame]:
    """Return a single-column DataFrame suitable for ``st.bar_chart``.

    Args:
        analysis: Loaded analysis dict.
        feature_type: ``"atom"`` or ``"bond"``.
        prop: The property enum value (AtomFeature or BondFeature).
        smiles_col: Which SMILES column to look up.

    Returns:
        DataFrame with a ``"Frequency"`` column and the property value as the
        index, or ``None`` if the data is absent.
    """
    col_data = analysis.get(feature_type, {}).get(str(prop), {}).get(smiles_col, {})
    bar_data = col_data.get("bar_data")
    if not bar_data:
        return None
    df_plot = pd.DataFrame(bar_data, columns=[str(prop), "Frequency"]).set_index(str(prop))
    return df_plot
