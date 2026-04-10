"""
polynet.featurizer.pmx
====================

Factory utilities for constructing PolyMetriX featurizers used in PolyNet.

This module provides a high-level helper to build a
:class:`polymetrix.featurizers.multiple_featurizer.MultipleFeaturizer`
from user-specified side-chain and backbone feature definitions.

Design
------
Feature definitions are split into two categories:

- **Chemical features**
  Molecular descriptors computed on a structure fragment, such as
  ring counts, molecular weight, or hydrogen-bond counts.

- **Topological features**
  Polymer-specific structural descriptors that are already implemented
  as complete featurizer classes, such as number of side chains or
  side-chain length.

Construction rules
------------------
- Chemical features requested for the side chain are wrapped as::

      SideChainFeaturizer(<chemical_feature>(agg=<agg_methods>))

- Chemical features requested for the backbone are wrapped as::

      BackBoneFeaturizer(<chemical_feature>())

- Topological features are instantiated directly and are not wrapped.

Public API
----------
::

    from polynet.features.pmx import create_pmx_featurizer

    featurizer = create_pmx_featurizer(
        side_chain_features=[PMXChemFeature.NumRings, PMXTopoFeature.NumSideChainFeaturizer],
        backbone_features=[PMXChemFeature.NumRings],
        agg_method=[PMXAggMethod.Sum, PMXAggMethod.Mean],
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

from polymetrix.featurizers.chemical_featurizer import (
    BalabanJIndex,
    BondCounts,
    BridgingRingsCount,
    FpDensityMorgan1,
    FractionBicyclicRings,
    HalogenCounts,
    HeteroatomCount,
    HeteroatomDensity,
    MaxEStateIndex,
    MaxRingSize,
    MolecularWeight,
    NumAliphaticHeterocycles,
    NumAromaticRings,
    NumAtoms,
    NumHBondAcceptors,
    NumHBondDonors,
    NumNonAromaticRings,
    NumRings,
    NumRotatableBonds,
    SlogPVSA1,
    SmrVSA5,
    Sp2CarbonCountFeaturizer,
    Sp3CarbonCountFeaturizer,
    TopologicalSurfaceArea,
)
from polymetrix.featurizers.multiple_featurizer import MultipleFeaturizer
from polymetrix.featurizers.sidechain_backbone_featurizer import (
    BackBoneFeaturizer,
    NumBackBoneFeaturizer,
    NumSideChainFeaturizer,
    SideChainFeaturizer,
    SidechainDiversityFeaturizer,
    SidechainLengthToStarAttachmentDistanceRatioFeaturizer,
    StarToSidechainMinDistanceFeaturizer,
)

from polynet.config.enums import PMXAggMethod, PMXChemFeature, PMXTopoFeature

PMXFeature: TypeAlias = PMXChemFeature | PMXTopoFeature | str
PMXAgg: TypeAlias = PMXAggMethod | str


CHEMICAL_FEATURIZER_REGISTRY = {
    PMXChemFeature.NumHBondDonors: NumHBondDonors,
    PMXChemFeature.NumHBondAcceptors: NumHBondAcceptors,
    PMXChemFeature.NumRotatableBonds: NumRotatableBonds,
    PMXChemFeature.NumRings: NumRings,
    PMXChemFeature.NumNonAromaticRings: NumNonAromaticRings,
    PMXChemFeature.NumAromaticRings: NumAromaticRings,
    PMXChemFeature.NumAtoms: NumAtoms,
    PMXChemFeature.TopologicalSurfaceArea: TopologicalSurfaceArea,
    PMXChemFeature.FractionBicyclicRings: FractionBicyclicRings,
    PMXChemFeature.NumAliphaticHeterocycles: NumAliphaticHeterocycles,
    PMXChemFeature.SlogPVSA1: SlogPVSA1,
    PMXChemFeature.BalabanJIndex: BalabanJIndex,
    PMXChemFeature.MolecularWeight: MolecularWeight,
    PMXChemFeature.Sp3CarbonCountFeaturizer: Sp3CarbonCountFeaturizer,
    PMXChemFeature.Sp2CarbonCountFeaturizer: Sp2CarbonCountFeaturizer,
    PMXChemFeature.MaxEStateIndex: MaxEStateIndex,
    PMXChemFeature.SmrVSA5: SmrVSA5,
    PMXChemFeature.FpDensityMorgan1: FpDensityMorgan1,
    PMXChemFeature.HalogenCounts: HalogenCounts,
    PMXChemFeature.BondCounts: BondCounts,
    PMXChemFeature.BridgingRingsCount: BridgingRingsCount,
    PMXChemFeature.MaxRingSize: MaxRingSize,
    PMXChemFeature.HeteroatomCount: HeteroatomCount,
    PMXChemFeature.HeteroatomDensity: HeteroatomDensity,
}

SIDECHAIN_TOPOLOGICAL_FEATURIZER_REGISTRY = {
    PMXTopoFeature.NumSideChainFeaturizer: NumSideChainFeaturizer,
    PMXTopoFeature.SidechainLengthToStarAttachmentDistanceRatioFeaturizer: SidechainLengthToStarAttachmentDistanceRatioFeaturizer,
    PMXTopoFeature.StarToSidechainMinDistanceFeaturizer: StarToSidechainMinDistanceFeaturizer,
    PMXTopoFeature.SidechainDiversityFeaturizer: SidechainDiversityFeaturizer,
}

BACKBONE_TOPOLOGICAL_FEATURIZER_REGISTRY = {
    PMXTopoFeature.NumBackBoneFeaturizer: NumBackBoneFeaturizer,
}

# Combined registry used by the factory — preserves backwards compatibility.
TOPOLOGICAL_FEATURIZER_REGISTRY = {
    **SIDECHAIN_TOPOLOGICAL_FEATURIZER_REGISTRY,
    **BACKBONE_TOPOLOGICAL_FEATURIZER_REGISTRY,
}


def _parse_chem_feature(feature: PMXChemFeature | str) -> PMXChemFeature:
    """Parse a chemical feature enum from enum or string input."""
    return PMXChemFeature(feature) if isinstance(feature, str) else feature


def _parse_topo_feature(feature: PMXTopoFeature | str) -> PMXTopoFeature:
    """Parse a topological feature enum from enum or string input."""
    return PMXTopoFeature(feature) if isinstance(feature, str) else feature


def _parse_agg_methods(agg_method: Sequence[PMXAgg]) -> list[str]:
    """
    Normalize aggregation methods to the string values expected by PolyMetriX.

    Parameters
    ----------
    agg_method:
        Sequence of aggregation methods as enum members or strings.

    Returns
    -------
    list[str]
        Normalized aggregation method names.
    """
    return [m.value if isinstance(m, PMXAggMethod) else str(m) for m in agg_method]


def _is_chemical_feature(feature: PMXFeature) -> bool:
    """Return True if the feature belongs to the chemical feature enum."""
    try:
        _parse_chem_feature(feature)
        return True
    except ValueError:
        return False


def _is_topological_feature(feature: PMXFeature) -> bool:
    """Return True if the feature belongs to the topological feature enum."""
    try:
        _parse_topo_feature(feature)
        return True
    except ValueError:
        return False


def _validate_features(
    side_chain_features: Sequence[PMXFeature], backbone_features: Sequence[PMXFeature]
) -> None:
    """
    Validate that all requested features are known.

    Raises
    ------
    ValueError
        If any requested feature is not a valid PMX chemical or topological feature.
    """
    unknown: list[str] = []

    for feature in list(side_chain_features) + list(backbone_features):
        if not (_is_chemical_feature(feature) or _is_topological_feature(feature)):
            unknown.append(str(feature))

    if unknown:
        raise ValueError(
            f"Unsupported PMX feature(s): {sorted(unknown)}. "
            f"Supported chemical features: {[f.value for f in PMXChemFeature]}. "
            f"Supported topological features: {[f.value for f in PMXTopoFeature]}."
        )


def _build_side_chain_chemical_features(
    features: Sequence[PMXFeature], agg_method: Sequence[PMXAgg]
) -> list[object]:
    """
    Build side-chain-wrapped chemical featurizers.

    Each chemical feature is instantiated with the provided aggregation methods
    and wrapped in :class:`SideChainFeaturizer`.
    """
    agg = _parse_agg_methods(agg_method)
    built: list[object] = []

    for feature in features:
        if _is_chemical_feature(feature):
            parsed = _parse_chem_feature(feature)
            feature_cls = CHEMICAL_FEATURIZER_REGISTRY[parsed]
            built.append(SideChainFeaturizer(feature_cls(agg=agg)))

    return built


def _build_backbone_chemical_features(features: Sequence[PMXFeature]) -> list[object]:
    """
    Build backbone-wrapped chemical featurizers.

    Each chemical feature is instantiated without aggregation and wrapped in
    :class:`BackBoneFeaturizer`.
    """
    built: list[object] = []

    for feature in features:
        if _is_chemical_feature(feature):
            parsed = _parse_chem_feature(feature)
            feature_cls = CHEMICAL_FEATURIZER_REGISTRY[parsed]
            built.append(BackBoneFeaturizer(feature_cls()))

    return built


def _build_topological_features(features: Sequence[PMXFeature]) -> list[object]:
    """
    Build direct topological featurizers.

    Topological features are already full featurizer classes and are therefore
    instantiated directly without side-chain/backbone wrapping.
    """
    built: list[object] = []

    for feature in features:
        if _is_topological_feature(feature):
            parsed = _parse_topo_feature(feature)
            feature_cls = TOPOLOGICAL_FEATURIZER_REGISTRY[parsed]
            built.append(feature_cls())

    return built


def create_pmx_featurizer(
    side_chain_features: Sequence[PMXFeature],
    backbone_features: Sequence[PMXFeature],
    agg_method: Sequence[PMXAgg],
) -> MultipleFeaturizer:
    """
    Create a PolyMetriX multiple featurizer from side-chain and backbone feature definitions.

    Parameters
    ----------
    side_chain_features:
        Features to compute on side chains. May include both chemical features
        (which are wrapped with :class:`SideChainFeaturizer`) and topological
        features (which are instantiated directly).
    backbone_features:
        Features to compute on the polymer backbone. Chemical features are wrapped
        with :class:`BackBoneFeaturizer`; topological features are instantiated directly.
    agg_method:
        Aggregation methods passed to side-chain chemical featurizers, e.g.
        ``["sum", "mean"]`` or ``[PMXAggMethod.Sum, PMXAggMethod.Mean]``.

    Returns
    -------
    MultipleFeaturizer
        Composite featurizer containing all requested features.

    Raises
    ------
    ValueError
        If any requested feature is not supported.

    Examples
    --------
    >>> featurizer = create_pmx_featurizer(
    ...     side_chain_features=[
    ...         PMXChemFeature.NumRings,
    ...         PMXTopoFeature.NumSideChainFeaturizer,
    ...     ],
    ...     backbone_features=[PMXChemFeature.NumRings],
    ...     agg_method=[PMXAggMethod.Sum, PMXAggMethod.Mean],
    ... )
    """
    _validate_features(side_chain_features=side_chain_features, backbone_features=backbone_features)

    side_chain_chemical = _build_side_chain_chemical_features(
        features=side_chain_features, agg_method=agg_method
    )
    backbone_chemical = _build_backbone_chemical_features(backbone_features)
    topological = _build_topological_features(list(side_chain_features) + list(backbone_features))

    all_features = side_chain_chemical + backbone_chemical + topological
    return MultipleFeaturizer(all_features)


# Backwards-compatible alias
PMX_featurizer_creator = create_pmx_featurizer
