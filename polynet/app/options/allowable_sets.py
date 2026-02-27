from rdkit.Chem.rdchem import BondStereo, HybridizationType

from polynet.config.enums import AtomBondDescriptorDictKey, AtomFeature, BondFeature

# atomic_properties = [
#     "GetAtomicNum",
#     # "GetSymbol",
#     # "GetDegree",
#     "GetTotalDegree",
#     "GetFormalCharge",
#     # "GetNumExplicitHs",
#     # "GetNumImplicitHs",
#     "GetTotalNumHs",
#     "GetIsAromatic",
#     "GetMass",
#     "GetHybridization",
#     "GetChiralTag",
#     # "GetImplicitValence",
#     # "GetExplicitValence",
#     # "GetTotalValence",
#     # "GetNumRadicalElectrons",
#     "IsInRing",
#     # "IsInRingSize",
# ]

# bond_properties = [
#     "GetBondTypeAsDouble",
#     "GetIsAromatic",
#     "GetIsConjugated",
#     "GetStereo",
#     "IsInRing",
#     "IsInRingSize",
# ]

atom_properties = {}

atom_properties[AtomFeature.GetAtomicNum] = {
    AtomBondDescriptorDictKey.Options: list(range(1, 85)),
    AtomBondDescriptorDictKey.Default: list(range(1, 37)) + [53],
}

atom_properties[AtomFeature.GetTotalDegree] = {
    AtomBondDescriptorDictKey.Options: list(range(0, 7)),
    AtomBondDescriptorDictKey.Default: list(range(6)),
}

atom_properties[AtomFeature.GetFormalCharge] = {
    AtomBondDescriptorDictKey.Options: list(range(-3, 3)),
    AtomBondDescriptorDictKey.Default: list(range(-2, 2)),
}

atom_properties[AtomFeature.GetTotalNumHs] = {
    AtomBondDescriptorDictKey.Options: list(range(0, 6)),
    AtomBondDescriptorDictKey.Default: list(range(5)),
}

atom_properties[AtomFeature.GetHybridization] = {
    AtomBondDescriptorDictKey.Options: [
        HybridizationType.S,
        HybridizationType.SP,
        HybridizationType.SP2,
        HybridizationType.SP2D,
        HybridizationType.SP3,
        HybridizationType.SP3D,
        HybridizationType.SP3D2,
    ],
    AtomBondDescriptorDictKey.Default: [
        HybridizationType.S,
        HybridizationType.SP,
        HybridizationType.SP2,
        HybridizationType.SP2D,
        HybridizationType.SP3,
        HybridizationType.SP3D,
        HybridizationType.SP3D2,
    ],
}

atom_properties[AtomFeature.GetChiralTag] = {
    AtomBondDescriptorDictKey.Options: [0, 1, 2, 3],
    AtomBondDescriptorDictKey.Default: list(range(4)),
}

atom_properties[AtomFeature.GetIsAromatic] = None  # This is a boolean property, no options needed
atom_properties[AtomFeature.GetMass] = None  # This is a continuous property, no options needed

atom_properties[AtomFeature.GetImplicitValence] = {
    AtomBondDescriptorDictKey.Options: list(range(6)),
    AtomBondDescriptorDictKey.Default: list(range(6)),
}

atom_properties[AtomFeature.IsInRing] = None  # This is a boolean property, no options needed


bond_features = {}
bond_features[BondFeature.GetBondTypeAsDouble] = {
    AtomBondDescriptorDictKey.Options: [1.0, 1.5, 2.0, 3.0],
    AtomBondDescriptorDictKey.Default: [1.0, 1.5, 2.0, 3.0],
}
bond_features[BondFeature.GetIsAromatic] = None  # This is a boolean property, no options needed
bond_features[BondFeature.GetIsConjugated] = None  # This is a boolean property, no options needed
bond_features[BondFeature.GetStereo] = {
    AtomBondDescriptorDictKey.Options: [BondStereo.STEREOE, BondStereo.STEREOZ],
    AtomBondDescriptorDictKey.Default: [BondStereo.STEREOE, BondStereo.STEREOZ],
}  # 0: NONE, 1: UP, 2: DOWN, 3: EITHER
bond_features[BondFeature.IsInRing] = None  # This is a boolean property, no options needed
