import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from polynet.featurizer.graph_representation.polymer_graph import PolymerGraphDataset
from polynet.options.enums import AtomBondDescriptorDictKeys, AtomFeatures, BondFeatures


class CustomPolymerGraph(PolymerGraphDataset):
    def __init__(
        self,
        filename: str = None,
        root: Path = None,
        smiles_cols: list = None,
        target_col: str = None,
        id_col: str = None,
        weights_col: dict = None,
        node_feats: dict = None,
        edge_feats: dict = None,
    ):
        self.name = "CustomPolymerGraph"
        self.weights_col = weights_col
        self.node_feats = node_feats
        self.edge_feats = edge_feats

        super().__init__(
            filename=filename,
            root=root,
            smiles_col=smiles_cols,
            target_col=target_col,
            id_col=id_col,
        )

    def process(self):
        smiles_col = self.smiles_col
        target_col = self.target_col
        id_col = self.id_col

        data = pd.read_csv(self.raw_paths[0])

        for index, mols in tqdm(data.iterrows(), total=data.shape[0]):
            monomers = []
            node_feats_polymer = None

            for monomer in smiles_col:
                smiles = mols[monomer]

                monomers.append(smiles)

                mol = Chem.MolFromSmiles(smiles)

                if mol is None:
                    print(f"Error in processing molecule {smiles}")
                    continue

                # Add node features
                node_feats = self._atom_features(mol)
                edge_index, edge_features = self._get_edge_idx(mol)

                if node_feats_polymer is None:
                    node_feats_polymer = node_feats
                    edge_index_polymer = edge_index
                    edge_feats_polymer = edge_features

                    # Check if weights_col is provided and set the weight_monomer
                    if self.weights_col:
                        weight_monomer = torch.full(
                            (node_feats.shape[0], 1), mols[self.weights_col[monomer]] / 100
                        )

                else:
                    node_feats_polymer = torch.cat((node_feats_polymer, node_feats), axis=0)
                    edge_index += max(edge_index_polymer[0]) + 1
                    edge_index_polymer = torch.cat((edge_index_polymer, edge_index), axis=1)
                    edge_feats_polymer = torch.cat((edge_feats_polymer, edge_features), axis=0)

                    if self.weights_col:
                        weight_monomer = torch.cat(
                            (
                                weight_monomer,
                                torch.full(
                                    (node_feats.shape[0], 1), mols[self.weights_col[monomer]] / 100
                                ),
                            )
                        )

            y = torch.tensor(mols[target_col], dtype=torch.float32)

            if id_col is not None:
                id = mols[id_col]
            else:
                id = index

            if not self.weights_col:
                weight_monomer = None

            data = Data(
                x=node_feats_polymer,
                edge_attr=edge_feats_polymer,
                edge_index=edge_index_polymer,
                y=y,
                weight_monomer=weight_monomer,
                mols=monomers,
                idx=id,
            )

            torch.save(
                data, os.path.join(self.processed_dir, f"{self.name}_{index}_{self.target_col}.pt")
            )

    def _atom_features(self, mol):

        mol_feats = []
        for a in mol.GetAtoms():
            node_feats = []

            # atomic number
            if AtomFeatures.GetAtomicNum in self.node_feats.keys():
                node_feats += self._one_h_e(
                    x=a.GetAtomicNum(),
                    allowable_set=self.node_feats[AtomFeatures.GetAtomicNum][
                        AtomBondDescriptorDictKeys.AllowableVals
                    ],
                    allow_low_frequency=self.node_feats[AtomFeatures.GetAtomicNum][
                        AtomBondDescriptorDictKeys.Wildcard
                    ],
                )

            # Num hydrogens
            if AtomFeatures.GetTotalNumHs in self.node_feats.keys():
                node_feats += self._one_h_e(
                    x=int(a.GetTotalNumHs()),
                    allowable_set=self.node_feats[AtomFeatures.GetTotalNumHs][
                        AtomBondDescriptorDictKeys.AllowableVals
                    ],
                    allow_low_frequency=self.node_feats[AtomFeatures.GetTotalNumHs][
                        AtomBondDescriptorDictKeys.Wildcard
                    ],
                )

            # Degree
            if AtomFeatures.GetTotalDegree in self.node_feats.keys():
                node_feats += self._one_h_e(
                    x=a.GetTotalDegree(),
                    allowable_set=self.node_feats[AtomFeatures.GetTotalDegree][
                        AtomBondDescriptorDictKeys.AllowableVals
                    ],
                    allow_low_frequency=self.node_feats[AtomFeatures.GetTotalDegree][
                        AtomBondDescriptorDictKeys.Wildcard
                    ],
                )

            # Implicit valence
            if AtomFeatures.GetImplicitValence in self.node_feats.keys():
                node_feats += self._one_h_e(
                    x=a.GetImplicitValence(),
                    allowable_set=self.node_feats[AtomFeatures.GetImplicitValence][
                        AtomBondDescriptorDictKeys.AllowableVals
                    ],
                    allow_low_frequency=self.node_feats[AtomFeatures.GetImplicitValence][
                        AtomBondDescriptorDictKeys.Wildcard
                    ],
                )

            # Aromaticity
            if AtomFeatures.GetIsAromatic in self.node_feats.keys():
                node_feats += [a.GetIsAromatic()]

            # Hybridization
            if AtomFeatures.GetHybridization in self.node_feats.keys():
                node_feats += self._one_h_e(
                    x=a.GetHybridization(),
                    allowable_set=self.node_feats[AtomFeatures.GetHybridization][
                        AtomBondDescriptorDictKeys.AllowableVals
                    ],
                    allow_low_frequency=self.node_feats[AtomFeatures.GetHybridization][
                        AtomBondDescriptorDictKeys.Wildcard
                    ],
                )

            # Formal charge
            if AtomFeatures.GetFormalCharge in self.node_feats.keys():
                node_feats += self._one_h_e(
                    x=a.GetFormalCharge(),
                    allowable_set=self.node_feats[AtomFeatures.GetFormalCharge][
                        AtomBondDescriptorDictKeys.AllowableVals
                    ],
                    allow_low_frequency=self.node_feats[AtomFeatures.GetFormalCharge][
                        AtomBondDescriptorDictKeys.Wildcard
                    ],
                )

            # Chirality
            if AtomFeatures.GetChiralTag in self.node_feats.keys():
                chirality = a.GetChiralTag()
                node_feats += self._one_h_e(
                    x=chirality,
                    allowable_set=self.node_feats[AtomFeatures.GetChiralTag][
                        AtomBondDescriptorDictKeys.AllowableVals
                    ],
                    allow_low_frequency=self.node_feats[AtomFeatures.GetChiralTag][
                        AtomBondDescriptorDictKeys.Wildcard
                    ],
                )

            # Mass
            if AtomFeatures.GetMass in self.node_feats.keys():
                # Get the mass of the atom and normalize it
                node_feats += [a.GetMass() / 100]

            # Is in ring
            if AtomFeatures.IsInRing in self.node_feats.keys():
                node_feats += [int(a.IsInRing())]

            mol_feats.append(node_feats)

        mol_feats = np.asarray(mol_feats, dtype=np.float32)
        return torch.tensor(mol_feats, dtype=torch.float)

    def _get_edge_idx(self, mol):
        mol_edge_feats = []
        edge_indices = []

        for bond in mol.GetBonds():
            bond_edge_feats = []
            # Feature 1: Bond type (as double)
            if BondFeatures.GetBondTypeAsDouble in self.edge_feats.keys():
                bond_edge_feats += self._one_h_e(
                    x=bond.GetBondTypeAsDouble(),
                    allowable_set=self.edge_feats[BondFeatures.GetBondTypeAsDouble][
                        AtomBondDescriptorDictKeys.AllowableVals
                    ],
                    allow_low_frequency=self.edge_feats[BondFeatures.GetBondTypeAsDouble][
                        AtomBondDescriptorDictKeys.Wildcard
                    ],
                )
            # Feature 2: Bond stereo
            if BondFeatures.GetStereo in self.edge_feats.keys():
                bond_edge_feats += self._one_h_e(
                    x=bond.GetStereo(),
                    allowable_set=self.edge_feats[BondFeatures.GetStereo][
                        AtomBondDescriptorDictKeys.AllowableVals
                    ],
                    allow_low_frequency=self.edge_feats[BondFeatures.GetStereo][
                        AtomBondDescriptorDictKeys.Wildcard
                    ],
                )
            # Feature 3: Is in ring
            if BondFeatures.IsInRing in self.edge_feats.keys():
                bond_edge_feats += [int(bond.IsInRing())]
            # Feature 4: Is conjugated
            if BondFeatures.GetIsConjugated in self.edge_feats.keys():
                bond_edge_feats += [int(bond.GetIsConjugated())]
            # Feature 5: Is aromatic
            if BondFeatures.GetIsAromatic in self.edge_feats.keys():
                bond_edge_feats += [int(bond.GetIsAromatic())]

            mol_edge_feats += [bond_edge_feats, bond_edge_feats]
            # Append edge indices to list (twice, per direction)
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # create adjacency list
            edge_indices += [[i, j], [j, i]]

        mol_edge_feats = torch.tensor(mol_edge_feats, dtype=torch.float)
        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)

        return edge_indices, mol_edge_feats
