"""
polynet.featurizer.polymer_graph
==================================
Concrete polymer graph dataset for custom node and edge feature configurations.

``CustomPolymerGraph`` builds molecular graphs from SMILES strings using
user-defined atom and bond feature sets. For multi-monomer polymers (multiple
SMILES columns), monomer graphs are concatenated into a single disconnected
graph with optional per-node monomer weight annotations.

Usage
-----
::

    from polynet.featurizer.polymer_graph import CustomPolymerGraph

    dataset = CustomPolymerGraph(
        filename="polymers.csv",
        root="data/processed/tg/",
        smiles_cols=["smiles_monomer_1", "smiles_monomer_2"],
        target_col="Tg",
        id_col="polymer_id",
        weights_col={"smiles_monomer_1": "ratio_1", "smiles_monomer_2": "ratio_2"},
        node_feats={
            AtomFeature.GetAtomicNum: {
                AtomBondDescriptorDictKey.AllowableVals: [6, 7, 8, 9],
                AtomBondDescriptorDictKey.Wildcard: True,
            },
        },
        edge_feats={
            BondFeature.GetBondTypeAsDouble: {
                AtomBondDescriptorDictKey.AllowableVals: [1.0, 1.5, 2.0],
                AtomBondDescriptorDictKey.Wildcard: False,
            },
        },
    )
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from tqdm import tqdm

from polynet.config.enums import AtomBondDescriptorDictKey, AtomFeature, BondFeature
from polynet.featurizer.graph import PolymerGraphDataset

logger = logging.getLogger(__name__)


node_feat_def = {
    "GetAtomicNum": {"allowable_vals": [8, 9, 6, 7], "wildcard": True},
    "GetTotalNumHs": {"allowable_vals": [0, 1, 2, 3], "wildcard": False},
    "GetTotalDegree": {"allowable_vals": [1, 2, 3, 4], "wildcard": False},
    "GetImplicitValence": {"allowable_vals": [0, 1, 2, 3], "wildcard": False},
    "GetIsAromatic": {},
}

edge_feat_def = {
    "GetBondTypeAsDouble": {"allowable_vals": [1.0, 2.0, 1.5], "wildcard": False},
    "GetIsAromatic": {},
}


class CustomPolymerGraph(PolymerGraphDataset):
    """
    Polymer graph dataset with user-defined atom and bond feature sets.

    Builds one PyG ``Data`` object per polymer. For multi-monomer polymers,
    monomer graphs are concatenated along the node and edge dimensions into
    a single disconnected graph. Monomer weight fractions are optionally
    stored as a per-node attribute (``weight_monomer``).

    Parameters
    ----------
    filename:
        Name of the raw CSV file (relative to ``root/raw/``).
    root:
        Root directory for raw and processed data storage.
    smiles_cols:
        Column names containing SMILES strings. One graph is built per
        column and concatenated into the final polymer graph.
    target_col:
        Target property column name. Pass ``None`` for inference datasets.
    id_col:
        Optional sample identifier column.
    weights_col:
        Optional mapping from SMILES column name to the column containing
        that monomer's weight fraction (as a percentage, 0–100).
    node_feats:
        Mapping from ``AtomFeature`` to its feature configuration dict.
        The dict must contain ``AtomBondDescriptorDictKey.AllowableVals``
        and ``AtomBondDescriptorDictKey.Wildcard`` keys.
    edge_feats:
        Mapping from ``BondFeature`` to its feature configuration dict,
        with the same structure as ``node_feats``.
    """

    name = "CustomPolymerGraph"

    def __init__(
        self,
        filename: str | None = None,
        root: str | Path | None = None,
        smiles_cols: list[str] | None = None,
        target_col: str | None = None,
        id_col: str | None = None,
        weights_col: dict[str, str] | None = None,
        node_feats: dict[AtomFeature, dict] | None = node_feat_def,
        edge_feats: dict[BondFeature, dict] | None = edge_feat_def,
    ) -> None:
        self.weights_col = weights_col
        self.node_feats = node_feats or {}
        self.edge_feats = edge_feats or {}

        super().__init__(
            filename=filename,
            root=str(root) if root is not None else root,
            smiles_col=smiles_cols,
            target_col=target_col,
            id_col=id_col,
        )

    # ------------------------------------------------------------------
    # PyG processing
    # ------------------------------------------------------------------

    def process(self) -> None:
        """
        Build and save one PyG ``Data`` object per row in the raw CSV.

        Each polymer is represented as a single graph formed by
        concatenating the individual monomer graphs. Graphs are saved
        as ``.pt`` files in ``root/processed/``.
        """
        data_df = pd.read_csv(self.raw_paths[0])

        for index, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Processing graphs"):
            graph = self._build_polymer_graph(row, data_df, index)
            if graph is None:
                continue
            torch.save(graph, self._graph_filepath(index))

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_polymer_graph(
        self, row: pd.Series, data_df: pd.DataFrame, index: int
    ) -> Data | None:
        """
        Build a single PyG ``Data`` object for one polymer row.

        Returns ``None`` if all monomers in the row fail to parse.
        """
        accumulated_node_feats: torch.Tensor | None = None
        accumulated_edge_index: torch.Tensor | None = None
        accumulated_edge_feats: torch.Tensor | None = None
        accumulated_weights: torch.Tensor | None = None
        monomers: list[str] = []

        for smiles_col in self.smiles_col:
            smiles = row[smiles_col]
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                logger.warning(
                    f"Could not parse SMILES '{smiles}' in column '{smiles_col}' "
                    f"(row {row.name}) — skipping monomer."
                )
                continue

            monomers.append(smiles)
            node_feats = self._atom_features(mol)
            edge_index, edge_feats = self._bond_features(mol)
            n_existing_nodes = (
                accumulated_node_feats.shape[0] if accumulated_node_feats is not None else 0
            )

            if accumulated_node_feats is None:
                accumulated_node_feats = node_feats
                accumulated_edge_index = edge_index
                accumulated_edge_feats = edge_feats
            else:
                # Offset new edge indices by the number of already-accumulated nodes
                # Using n_existing_nodes (not max index) is correct for disconnected graphs
                edge_index = edge_index + n_existing_nodes
                accumulated_node_feats = torch.cat([accumulated_node_feats, node_feats], dim=0)
                accumulated_edge_index = torch.cat([accumulated_edge_index, edge_index], dim=1)
                accumulated_edge_feats = torch.cat([accumulated_edge_feats, edge_feats], dim=0)

            if self.weights_col and smiles_col in self.weights_col:
                weight_val = row[self.weights_col[smiles_col]] / 100.0
                monomer_weights = torch.full((node_feats.shape[0], 1), weight_val)
                accumulated_weights = (
                    monomer_weights
                    if accumulated_weights is None
                    else torch.cat([accumulated_weights, monomer_weights], dim=0)
                )

        if accumulated_node_feats is None:
            logger.error(f"All monomers failed to parse for row {row.name} — skipping polymer.")
            return None

        y = None
        if self.target_col is not None and self.target_col in data_df.columns:
            y = torch.tensor(row[self.target_col], dtype=torch.float32)

        sample_id = row[self.id_col] if self.id_col is not None else index

        return Data(
            x=accumulated_node_feats,
            edge_attr=accumulated_edge_feats,
            edge_index=accumulated_edge_index,
            y=y,
            weight_monomer=accumulated_weights,
            mols=monomers,
            idx=sample_id,
        )

    # ------------------------------------------------------------------
    # Atom feature extraction
    # ------------------------------------------------------------------

    def _atom_features(self, mol) -> torch.Tensor:
        """
        Extract node feature vectors for all atoms in a molecule.

        Features are computed in the order they appear in ``self.node_feats``,
        which preserves the user-defined feature ordering.

        Parameters
        ----------
        mol:
            RDKit molecule object.

        Returns
        -------
        torch.Tensor
            Float tensor of shape ``(n_atoms, n_node_features)``.
        """
        mol_feats: list[list] = []

        for atom in mol.GetAtoms():
            atom_feat: list = []

            feature_extractors = {
                AtomFeature.GetAtomicNum: lambda a: self._one_hot_encode(
                    a.GetAtomicNum(),
                    self.node_feats[AtomFeature.GetAtomicNum][
                        AtomBondDescriptorDictKey.AllowableVals
                    ],
                    self.node_feats[AtomFeature.GetAtomicNum][AtomBondDescriptorDictKey.Wildcard],
                ),
                AtomFeature.GetTotalNumHs: lambda a: self._one_hot_encode(
                    int(a.GetTotalNumHs()),
                    self.node_feats[AtomFeature.GetTotalNumHs][
                        AtomBondDescriptorDictKey.AllowableVals
                    ],
                    self.node_feats[AtomFeature.GetTotalNumHs][AtomBondDescriptorDictKey.Wildcard],
                ),
                AtomFeature.GetTotalDegree: lambda a: self._one_hot_encode(
                    a.GetTotalDegree(),
                    self.node_feats[AtomFeature.GetTotalDegree][
                        AtomBondDescriptorDictKey.AllowableVals
                    ],
                    self.node_feats[AtomFeature.GetTotalDegree][AtomBondDescriptorDictKey.Wildcard],
                ),
                AtomFeature.GetImplicitValence: lambda a: self._one_hot_encode(
                    a.GetImplicitValence(),
                    self.node_feats[AtomFeature.GetImplicitValence][
                        AtomBondDescriptorDictKey.AllowableVals
                    ],
                    self.node_feats[AtomFeature.GetImplicitValence][
                        AtomBondDescriptorDictKey.Wildcard
                    ],
                ),
                AtomFeature.GetIsAromatic: lambda a: [int(a.GetIsAromatic())],
                AtomFeature.GetHybridization: lambda a: self._one_hot_encode(
                    a.GetHybridization(),
                    self.node_feats[AtomFeature.GetHybridization][
                        AtomBondDescriptorDictKey.AllowableVals
                    ],
                    self.node_feats[AtomFeature.GetHybridization][
                        AtomBondDescriptorDictKey.Wildcard
                    ],
                ),
                AtomFeature.GetFormalCharge: lambda a: self._one_hot_encode(
                    a.GetFormalCharge(),
                    self.node_feats[AtomFeature.GetFormalCharge][
                        AtomBondDescriptorDictKey.AllowableVals
                    ],
                    self.node_feats[AtomFeature.GetFormalCharge][
                        AtomBondDescriptorDictKey.Wildcard
                    ],
                ),
                AtomFeature.GetChiralTag: lambda a: self._one_hot_encode(
                    a.GetChiralTag(),
                    self.node_feats[AtomFeature.GetChiralTag][
                        AtomBondDescriptorDictKey.AllowableVals
                    ],
                    self.node_feats[AtomFeature.GetChiralTag][AtomBondDescriptorDictKey.Wildcard],
                ),
                AtomFeature.GetMass: lambda a: [a.GetMass() / 100.0],
                AtomFeature.IsInRing: lambda a: [int(a.IsInRing())],
            }

            for feature, extractor in feature_extractors.items():
                if feature in self.node_feats:
                    atom_feat += extractor(atom)

            mol_feats.append(atom_feat)

        return torch.tensor(np.asarray(mol_feats, dtype=np.float32), dtype=torch.float)

    # ------------------------------------------------------------------
    # Bond feature extraction
    # ------------------------------------------------------------------

    def _bond_features(self, mol) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract edge indices and edge feature vectors for all bonds.

        Each bond is represented twice (both directions) to produce an
        undirected graph in COO format, as expected by PyG.

        Parameters
        ----------
        mol:
            RDKit molecule object.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(edge_index, edge_features)`` where:
            - ``edge_index`` has shape ``(2, 2 * n_bonds)``
            - ``edge_features`` has shape ``(2 * n_bonds, n_edge_features)``
        """
        edge_indices: list[list[int]] = []
        edge_feats_list: list[list] = []

        for bond in mol.GetBonds():
            bond_feat: list = []

            if BondFeature.GetBondTypeAsDouble in self.edge_feats:
                bond_feat += self._one_hot_encode(
                    bond.GetBondTypeAsDouble(),
                    self.edge_feats[BondFeature.GetBondTypeAsDouble][
                        AtomBondDescriptorDictKey.AllowableVals
                    ],
                    self.edge_feats[BondFeature.GetBondTypeAsDouble][
                        AtomBondDescriptorDictKey.Wildcard
                    ],
                )
            if BondFeature.GetStereo in self.edge_feats:
                bond_feat += self._one_hot_encode(
                    bond.GetStereo(),
                    self.edge_feats[BondFeature.GetStereo][AtomBondDescriptorDictKey.AllowableVals],
                    self.edge_feats[BondFeature.GetStereo][AtomBondDescriptorDictKey.Wildcard],
                )
            if BondFeature.IsInRing in self.edge_feats:
                bond_feat += [int(bond.IsInRing())]
            if BondFeature.GetIsConjugated in self.edge_feats:
                bond_feat += [int(bond.GetIsConjugated())]
            if BondFeature.GetIsAromatic in self.edge_feats:
                bond_feat += [int(bond.GetIsAromatic())]

            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]
            edge_feats_list += [bond_feat, bond_feat]

        edge_index = torch.tensor(edge_indices).t().to(torch.long).view(2, -1)
        edge_feats = torch.tensor(edge_feats_list, dtype=torch.float)

        return edge_index, edge_feats
