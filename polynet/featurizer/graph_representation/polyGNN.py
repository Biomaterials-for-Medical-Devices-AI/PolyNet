import os

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from torch_geometric.data import Data
from tqdm import tqdm

from polynet.featurizer.graph_representation.polymer_graph import PolymerGraphDataset


class PolyGNN(PolymerGraphDataset):
    def __init__(
        self, root=None, filename=None, smiles_col=None, target_col=None, id_col=None
    ):
        self.name = "PolyGNN"

        super().__init__(
            root=root,
            filename=filename,
            smiles_col=smiles_col,
            target_col=target_col,
            id_col=id_col,
        )

    def process(
        self,
    ):
        smiles_col = self.smiles_col
        target_col = self.target_col
        id_col = self.id_col

        data = pd.read_csv(self.raw_paths[0])

        for index, mols in tqdm(data.iterrows(), total=data.shape[0]):
            smiles = Chem.MolToSmiles(
                Chem.MolFromSmiles(mols[smiles_col]), canonical=True
            )

            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                print(f"Error in processing molecule {smiles}")
                continue

            # Add node features
            node_feats = self._atom_features(mol)
            edge_index = self._get_edge_idx(mol)

            y = torch.tensor(mols[target_col], dtype=torch.float32)

            if id_col is not None:
                id = mols[id_col]
            else:
                id = index

            data = Data(
                x=node_feats,
                edge_attr=None,
                edge_index=edge_index,
                y=y,
                smiles=smiles,
                idx=id,
            )

            torch.save(
                data, os.path.join(self.processed_dir, f"{self.name}_{index}.pt")
            )

    def _atom_features(self, mol):
        sets = self.allowed_sets()
        mol_feats = []
        for a in mol.GetAtoms():
            node_feats = []
            # atomic number
            node_feats += self._one_h_e(
                a.GetAtomicNum(), sets["atomic_nums"], allow_low_frequency=True
            )
            # Num hydrogens
            node_feats += self._one_h_e(int(a.GetTotalNumHs()), sets["num_Hs"])
            # Degree
            node_feats += self._one_h_e(a.GetTotalDegree(), sets["degree"])
            # implicit valence
            node_feats += self._one_h_e(
                a.GetImplicitValence(), sets["implicit_valence"]
            )
            # aromaticity
            node_feats += [a.GetIsAromatic()]

            # node_feats += self._one_h_e(a.GetFormalCharge(), sets["formal_charge"])
            # node_feats += self._one_h_e(int(a.GetChiralTag()), sets["chirality"])
            # node_feats += self._one_h_e(int(a.GetTotalNumHs()), sets["num_Hs"])
            # node_feats += self._one_h_e(a.GetHybridization(), sets["hybridization"])
            # node_feats += [a.GetIsAromatic()]
            # node_feats += [a.GetMass() / 100]
            mol_feats.append(node_feats)

        mol_feats = np.asarray(mol_feats, dtype=np.float32)
        return torch.tensor(mol_feats, dtype=torch.float)

    def _get_edge_idx(self, mol):
        edge_indices = []

        for bond in mol.GetBonds():
            # Append edge indices to list (twice, per direction)
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # create adjacency list
            edge_indices += [[i, j], [j, i]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)

        return edge_indices

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
        }
        return sets
