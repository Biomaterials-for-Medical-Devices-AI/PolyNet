import os

import pandas as pd
import torch
from torch_geometric.data import Dataset


class PolymerGraphDataset(Dataset):
    def __init__(
        self,
        root=None,
        filename=None,
        smiles_col=None,
        target_col=None,
        id_col=None,
    ):
        self.name = "PolymerGraphDataset"
        self.filename = filename
        self.root = root
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.id_col = id_col
        super().__init__(root)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        molecules = [
            f"{self.name}_{i}_{self.target_col}.pt" for i in list(self.data.index)
        ]
        return molecules

    @property
    def _element_list(self):
        elements = [
            "H",  # Hydrogen (1)
            "Li",  # Lithium (3)
            "B",  # Boron (5)
            "C",  # Carbon (6)
            "N",  # Nitrogen (7)
            "O",  # Oxygen (8)
            "F",  # Fluorine (9)
            "Na",  # Sodium (11)
            "Si",  # Silicon (14)
            "P",  # Phosphorus (15)
            "S",  # Sulfur (16)
            "Cl",  # Chlorine (17)
            "K",  # Potassium (19)
            "Se",  # Selenium (34)
            "Br",  # Bromine (35)
            "I",  # Iodine (53)
        ]
        return elements

    def download(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    def _get_node_feats(self):
        raise NotImplementedError

    def _get_edge_features(self):
        raise NotImplementedError

    def _print_dataset_info(self) -> None:
        """
        Prints the dataset info
        """
        print(f"{self._name} dataset has {len(self)} samples")

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        molecule = torch.load(
            os.path.join(self.processed_dir, f"{self.name}_{idx}_{self.target_col}.pt"),
            weights_only=False,
        )
        return molecule

    def _get_atom_chirality(self, CIP_dict, atom_idx):
        try:
            chirality = CIP_dict[atom_idx]
        except KeyError:
            chirality = "No_Stereo_Center"

        return chirality

    def _one_h_e(self, x, allowable_set, ok_set=None, allow_low_frequency=False):
        vector = list(map(lambda s: x == s, allowable_set))  # Standard one-hot encoding

        if allow_low_frequency:
            vector.append(
                int(x not in allowable_set and (ok_set is None or x != ok_set))
            )  # Append 1 if x is not valid

        return vector
