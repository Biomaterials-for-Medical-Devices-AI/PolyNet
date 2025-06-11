import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from polynet.featurizer.graph_representation.polymer_graph import PolymerGraphDataset


class PolyChemprop(PolymerGraphDataset):
    def __init__(
        self,
        filename=None,
        root=None,
        smiles_col=None,
        target_col=None,
        id_col=None,
        ratio_col=None,
    ):
        self.name = "PolyChemProp"
        self.ratio_col = ratio_col
        self.sets = self.allowed_sets()

        super().__init__(
            filename=filename,
            root=root,
            smiles_col=smiles_col,
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
                smiles = Chem.MolToSmiles(Chem.MolFromSmiles(mols[monomer]), canonical=True)

                monomers.append(smiles)

                mol = Chem.MolFromSmiles(smiles)

                if mol is None:
                    print(f"Error in processing molecule {smiles}")
                    continue

                # Add node features
                node_feats = self._atom_features(mol)
                edge_index, edge_feats = self.get_edge_features(mol)

                if node_feats_polymer is None:
                    node_feats_polymer = node_feats
                    edge_index_polymer = edge_index
                    edge_feats_polymer = edge_feats
                    weight_monomer = torch.full(
                        (node_feats.shape[0], 1), mols[self.ratio_col] / 100
                    )

                else:
                    node_feats_polymer = torch.cat((node_feats_polymer, node_feats), axis=0)
                    edge_index += max(edge_index_polymer[0]) + 1
                    edge_index_polymer = torch.cat((edge_index_polymer, edge_index), axis=1)
                    edge_feats_polymer = torch.cat((edge_feats_polymer, edge_feats), axis=0)
                    weight_monomer = torch.cat(
                        (
                            weight_monomer,
                            torch.full((node_feats.shape[0], 1), 1 - mols[self.ratio_col] / 100),
                        )
                    )

            y = torch.tensor(mols[target_col], dtype=torch.float32)

            if id_col is not None:
                id = mols[id_col]
            else:
                id = index

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
            node_feats += self._one_h_e(
                a.GetAtomicNum(), self.sets["atomic_nums"], allow_low_frequency=True
            )
            # Degree
            node_feats += self._one_h_e(a.GetTotalDegree(), self.sets["degree"])
            # Formal charge
            node_feats += self._one_h_e(a.GetFormalCharge(), self.sets["formal_charge"])
            # chirality
            node_feats += self._one_h_e(int(a.GetChiralTag()), self.sets["chirality"])
            # Num hydrogens
            node_feats += self._one_h_e(int(a.GetTotalNumHs()), self.sets["num_Hs"])
            # Hybridization
            node_feats += self._one_h_e(a.GetHybridization(), self.sets["hybridization"])

            # aromaticity
            node_feats += [a.GetIsAromatic()]

            # Atomic mass
            node_feats += [a.GetMass() / 100]

            mol_feats.append(node_feats)

        mol_feats = np.asarray(mol_feats, dtype=np.float32)
        return torch.tensor(mol_feats, dtype=torch.float)

    def get_edge_features(self, mol):

        mol_edge_feats = []
        edges_indices = []

        for bond in mol.GetBonds():

            bond_edge_feats = []

            # Feature 1: Bond type (as double)
            bond_edge_feats += self._one_h_e(bond.GetBondType(), self.sets["bond_type"])
            # Feature 2: Bond stereo
            bond_edge_feats += self._one_h_e(bond.GetBondTypeAsDouble(), self.sets["bond_stereo"])
            # Feature 3: Is in ring
            bond_edge_feats += [bond.IsInRing()]
            # Feature 4: Is conjugated
            bond_edge_feats += [bond.GetIsConjugated()]

            mol_edge_feats += [bond_edge_feats, bond_edge_feats]

            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edges_indices += [[i, j], [j, i]]

        mol_edge_feats = torch.tensor(mol_edge_feats, dtype=torch.float)
        edges_indices = torch.tensor(edges_indices).t().to(torch.long).view(2, -1)
        return edges_indices, mol_edge_feats

    def allowed_sets(self):
        sets = {
            "atomic_nums": list(range(1, 37)) + [53],
            "num_Hs": list(range(5)),
            "degree": list(range(6)),
            "implicit_valence": list(range(6)),
            "formal_charge": list(range(-2, 2)),
            "chirality": list(range(4)),
            "hybridization": [
                HybridizationType.S,
                HybridizationType.SP,
                HybridizationType.SP2,
                HybridizationType.SP2D,
                HybridizationType.SP3,
                HybridizationType.SP3D,
                HybridizationType.SP3D2,
            ],
            "bond_type": [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC,
                Chem.rdchem.BondType.DATIVE,
            ],
            "bond_stereo": [Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE],
            "bond_in_ring": [int],
            "bond_conjugated": [int],
        }
        return sets
