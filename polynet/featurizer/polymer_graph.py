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
from rdkit import Chem
import torch
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
        that monomer's molar ratio. Ratios are normalised per polymer to sum
        to 1 across the participating monomers, so any scale works (fractions,
        percentages, or arbitrary ratios such as 1:1:2); a value of 0 excludes
        that monomer from the graph.
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
        polymer_descriptors: list[str] | None = None,
    ) -> None:
        self.weights_col = weights_col
        self.node_feats = node_feats or {}
        self.edge_feats = edge_feats or {}
        self.polymer_descriptors = polymer_descriptors or []

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
        accumulated_monomer_id: torch.Tensor | None = None
        weight_sum: float = 0.0
        monomers: list[str] = []

        for smiles_col in self.smiles_col:
            if self.weights_col and smiles_col in self.weights_col:
                if row[self.weights_col[smiles_col]] == 0:
                    continue

            smiles = row[smiles_col]
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                logger.warning(
                    f"Could not parse SMILES '{smiles}' in column '{smiles_col}' "
                    f"(row {row.name}) — skipping monomer."
                )
                continue

            monomers.append(smiles)

            # Opt-in: when ``IsAttachmentPoint`` is among the configured node
            # features, drop all wildcard ('*', atomic-num 0) atoms from the
            # molecule and instead flag their non-wildcard neighbours. Both
            # the stripping and the flagging are gated on the same switch so
            # they cannot get out of sync. When the feature is absent the
            # mol passes through unchanged (no behaviour change for existing
            # experiments).
            attachment_idxs: set[int] | None = None
            if AtomFeature.IsAttachmentPoint in self.node_feats:
                mol, attachment_idxs = self._strip_wildcards(mol)

            node_feats = self._atom_features(mol, attachment_idxs=attachment_idxs)
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
                # Store the raw molar ratio per node here; the per-polymer
                # normalisation (divide by the sum of the participating
                # monomers' ratios) is applied once after the loop. For ratios
                # that already sum to 100 this reproduces the previous
                # ``ratio / 100`` behaviour exactly; for any other scale
                # (fractions, 1:1:2, …) it yields per-monomer weights that sum
                # to 1 across the participating monomers.
                ratio_val = float(row[self.weights_col[smiles_col]])
                weight_sum += ratio_val
                monomer_weights = torch.full((node_feats.shape[0], 1), ratio_val)
                accumulated_weights = (
                    monomer_weights
                    if accumulated_weights is None
                    else torch.cat([accumulated_weights, monomer_weights], dim=0)
                )

            # Per-node local monomer index (0, 1, …) within this polymer.
            # ``len(monomers) - 1`` is the index of the monomer just appended;
            # skipped (zero-weight / unparseable) monomers are excluded, so ids
            # stay contiguous. Used by PerMonomerPooling to segment the graph.
            monomer_id = torch.full((node_feats.shape[0], 1), len(monomers) - 1, dtype=torch.long)
            accumulated_monomer_id = (
                monomer_id
                if accumulated_monomer_id is None
                else torch.cat([accumulated_monomer_id, monomer_id], dim=0)
            )

        if accumulated_node_feats is None:
            logger.error(f"All monomers failed to parse for row {row.name} — skipping polymer.")
            return None

        # Normalise the per-node monomer weights so each polymer's ratios sum to
        # 1 across its participating monomers. ``weight_sum`` is the sum of the
        # raw ratios of exactly the monomers that contributed nodes (zero-ratio
        # and unparseable monomers were skipped above), so this is the correct
        # denominator. A zero/negative sum is degenerate and left unnormalised.
        if accumulated_weights is not None and weight_sum > 0:
            accumulated_weights = accumulated_weights / weight_sum

        y = None
        if self.target_col is not None and self.target_col in data_df.columns:
            y = torch.tensor(row[self.target_col], dtype=torch.float32)

        sample_id = row[self.id_col] if self.id_col is not None else index

        poly_desc_tensor = None
        if self.polymer_descriptors:
            desc_vals = [row[col] for col in self.polymer_descriptors]
            poly_desc_tensor = torch.tensor([desc_vals], dtype=torch.float32)

        return Data(
            x=accumulated_node_feats,
            edge_attr=accumulated_edge_feats,
            edge_index=accumulated_edge_index,
            y=y,
            weight_monomer=accumulated_weights,
            monomer_id=accumulated_monomer_id,
            polymer_descriptors=poly_desc_tensor,
            mols=monomers,
            idx=sample_id,
        )

    # ------------------------------------------------------------------
    # Atom feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_wildcards(mol):
        """
        Remove wildcard atoms (atomic number 0, the ``*`` placeholders in
        PSMILES) from a molecule and return the stripped mol together with
        the set of *post-strip* atom indices that were directly bonded to a
        removed wildcard — i.e. the attachment points.

        Returns
        -------
        tuple[rdkit.Chem.Mol, set[int]]
            ``(stripped_mol, attachment_idxs_in_stripped_mol)``. When the
            input has no wildcard atoms the mol is returned unchanged and
            the set is empty (so non-PSMILES inputs degrade to an all-zero
            attachment-point column).
        """
        dummy_orig = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0}
        if not dummy_orig:
            return mol, set()

        attachment_orig = {
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetAtomicNum() != 0 and any(n.GetAtomicNum() == 0 for n in a.GetNeighbors())
        }

        # Removing atoms in descending index order keeps earlier indices stable
        # so we can rebuild a deterministic original -> stripped index map.
        rw = Chem.RWMol(mol)
        for idx in sorted(dummy_orig, reverse=True):
            rw.RemoveAtom(idx)
        stripped = rw.GetMol()

        kept_in_order = [i for i in range(mol.GetNumAtoms()) if i not in dummy_orig]
        orig_to_new = {orig: new for new, orig in enumerate(kept_in_order)}
        attachment_new = {orig_to_new[i] for i in attachment_orig}

        return stripped, attachment_new

    def _atom_features(self, mol, attachment_idxs: set[int] | None = None) -> torch.Tensor:
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
                    a.GetValence(Chem.rdchem.ValenceType.IMPLICIT),
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
                AtomFeature.IsAttachmentPoint: lambda a: [
                    int(attachment_idxs is not None and a.GetIdx() in attachment_idxs)
                ],
            }

            for feature, extractor in feature_extractors.items():
                if feature in self.node_feats:
                    atom_feat += extractor(atom)

            mol_feats.append(atom_feat)

        return torch.tensor(np.asarray(mol_feats, dtype=np.float32), dtype=torch.float)

    # ------------------------------------------------------------------
    # Bond feature extraction
    # ------------------------------------------------------------------

    def _n_edge_features(self) -> int:
        """
        Number of features produced per bond by :meth:`_bond_features`.

        Derived from the configured edge features so that bond-free molecules
        (e.g. a PSMILES monomer that reduces to a single atom after wildcard
        stripping) can return a correctly shaped ``(0, n_edge_features)`` tensor
        instead of a malformed 1-D ``(0,)`` tensor.
        """
        n = 0
        for feat in (BondFeature.GetBondTypeAsDouble, BondFeature.GetStereo):
            if feat in self.edge_feats:
                cfg = self.edge_feats[feat]
                n += len(cfg[AtomBondDescriptorDictKey.AllowableVals])
                n += int(bool(cfg[AtomBondDescriptorDictKey.Wildcard]))
        for feat in (BondFeature.IsInRing, BondFeature.GetIsConjugated, BondFeature.GetIsAromatic):
            if feat in self.edge_feats:
                n += 1
        return n

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
        if edge_feats_list:
            edge_feats = torch.tensor(edge_feats_list, dtype=torch.float)
        else:
            # No bonds (e.g. a PSMILES monomer that reduces to a single atom
            # after wildcard stripping). Build a correctly shaped empty tensor so
            # the edge-feature dimension is preserved for batching and for
            # edge-aware convolutions such as GATv2Conv.
            edge_feats = torch.zeros((0, self._n_edge_features()), dtype=torch.float)

        return edge_index, edge_feats
