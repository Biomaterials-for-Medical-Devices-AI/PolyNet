from rdkit.Chem.rdchem import HybridizationType
from polynet.options.enums import AtomFeatures, BondFeatures, AtomBondDescriptorDictKeys


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

atom_properties[AtomFeatures.GetAtomicNum] = {
    AtomBondDescriptorDictKeys.Options: list(range(1, 85)),
    AtomBondDescriptorDictKeys.Default: list(range(1, 37)) + [53],
}

atom_properties[AtomFeatures.GetTotalDegree] = {
    AtomBondDescriptorDictKeys.Options: list(range(0, 7)),
    AtomBondDescriptorDictKeys.Default: list(range(5)),
}

atom_properties[AtomFeatures.GetFormalCharge] = {
    AtomBondDescriptorDictKeys.Options: list(range(-3, 3)),
    AtomBondDescriptorDictKeys.Default: list(range(-2, 2)),
}

atom_properties[AtomFeatures.GetTotalNumHs] = {
    AtomBondDescriptorDictKeys.Options: list(range(0, 6)),
    AtomBondDescriptorDictKeys.Default: list(range(5)),
}

atom_properties[AtomFeatures.GetHybridization] = {
    AtomBondDescriptorDictKeys.Options: [
        HybridizationType.S,
        HybridizationType.SP,
        HybridizationType.SP2,
        HybridizationType.SP2D,
        HybridizationType.SP3,
        HybridizationType.SP3D,
        HybridizationType.SP3D2,
    ],
    AtomBondDescriptorDictKeys.Default: [
        HybridizationType.S,
        HybridizationType.SP,
        HybridizationType.SP2,
        HybridizationType.SP2D,
        HybridizationType.SP3,
        HybridizationType.SP3D,
        HybridizationType.SP3D2,
    ],
}

atom_properties[AtomFeatures.GetChiralTag] = {
    AtomBondDescriptorDictKeys.Options: [0, 1, 2, 3],
    AtomBondDescriptorDictKeys.Default: list(range(4)),
}

atom_properties[AtomFeatures.GetIsAromatic] = None  # This is a boolean property, no options needed
atom_properties[AtomFeatures.GetMass] = None  # This is a continuous property, no options needed

atom_properties[AtomFeatures.GetImplicitValence] = {
    AtomBondDescriptorDictKeys.Options: list(range(6)),
    AtomBondDescriptorDictKeys.Default: list(range(6)),
}

atom_properties[AtomFeatures.IsInRing] = None  # This is a boolean property, no options needed


bond_features = {}
bond_features[BondFeatures.GetBondTypeAsDouble] = {
    AtomBondDescriptorDictKeys.Options: [1.0, 1.5, 2.0, 3.0],
    AtomBondDescriptorDictKeys.Default: [1.0, 1.5],
}
bond_features[BondFeatures.GetIsAromatic] = None  # This is a boolean property, no options needed
bond_features[BondFeatures.GetIsConjugated] = None  # This is a boolean property, no options needed
bond_features[BondFeatures.GetStereo] = {
    AtomBondDescriptorDictKeys.Options: [0, 1, 2, 3],
    AtomBondDescriptorDictKeys.Default: [0, 1, 2],
}  # 0: NONE, 1: UP, 2: DOWN, 3: EITHER
bond_features[BondFeatures.IsInRing] = None  # This is a boolean property, no options needed
