"""
polynet.featurizer.graph
=========================
Base class for polymer graph datasets built on top of PyTorch Geometric.

``PolymerGraphDataset`` provides the shared scaffolding for loading,
processing, and serving polymer molecular graphs. Subclasses implement
``process()`` to define how raw data is converted into ``torch_geometric.data.Data``
objects.

Subclassing
-----------
To create a custom graph dataset, inherit from ``PolymerGraphDataset``
and implement ``process()``::

    from polynet.featurizer.graph import PolymerGraphDataset

    class MyDataset(PolymerGraphDataset):
        def process(self):
            ...

The ``_one_hot_encode`` and ``_element_list`` helpers are available to
all subclasses.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
import torch
from torch_geometric.data import Dataset

logger = logging.getLogger(__name__)


class PolymerGraphDataset(Dataset):
    """
    Base PyTorch Geometric Dataset for polymer molecular graphs.

    Handles file path resolution, processed file naming, and graph
    retrieval. Subclasses must implement ``process()`` to define how
    raw CSV data is converted into saved ``.pt`` graph files.

    Parameters
    ----------
    root:
        Root directory where raw and processed data are stored.
        PyG expects ``root/raw/`` and ``root/processed/`` subdirectories.
    filename:
        Name of the raw CSV file (relative to ``root/raw/``).
    smiles_col:
        One or more column names containing SMILES strings.
    target_col:
        Column name of the target property. Can be ``None`` for inference.
    id_col:
        Optional sample identifier column.
    """

    def __init__(
        self,
        root: str | None = None,
        filename: str | None = None,
        smiles_col: list[str] | str | None = None,
        target_col: str | None = None,
        id_col: str | None = None,
    ) -> None:
        self.filename = filename
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.id_col = id_col

        # Cache for processed file names — avoids repeated CSV reads
        self._processed_file_names_cache: list[str] | None = None

        super().__init__(root)

    # ------------------------------------------------------------------
    # PyG interface
    # ------------------------------------------------------------------

    @property
    def raw_file_names(self) -> list[str]:
        """List of raw files expected by PyG in ``root/raw/``."""
        return [self.filename] if self.filename else []

    @property
    def processed_file_names(self) -> list[str]:
        """
        List of processed ``.pt`` files expected by PyG in ``root/processed/``.

        Cached after the first call to avoid repeated CSV reads, since
        PyG calls this property multiple times during dataset initialisation.
        """
        if self._processed_file_names_cache is None:
            data = pd.read_csv(self.raw_paths[0]).reset_index()
            self._processed_file_names_cache = [self._graph_filename(i) for i in data.index]
        return self._processed_file_names_cache

    def download(self) -> None:
        """
        Not implemented — polynet datasets are loaded from local files.

        If you need to download data, subclass this method.
        """
        raise NotImplementedError(
            "Automatic download is not supported. "
            "Place the raw CSV file in the 'raw/' subdirectory of the dataset root."
        )

    def process(self) -> None:
        """
        Convert raw data into processed graph files.

        Must be implemented by subclasses. Each graph should be saved
        using ``torch.save(data, path)`` where ``path`` is obtained from
        ``self._graph_filepath(index)``.
        """
        raise NotImplementedError(
            "Subclasses must implement process() to define graph construction."
        )

    def len(self) -> int:
        """Number of graphs in the dataset."""
        return len(self.processed_file_names)

    def get(self, idx: int) -> object:
        """Load and return the graph at position ``idx``."""
        path = os.path.join(self.processed_dir, self._graph_filename(idx))
        return torch.load(path, weights_only=False)

    # ------------------------------------------------------------------
    # Subclass helpers
    # ------------------------------------------------------------------

    def _graph_filename(self, idx: int) -> str:
        """
        Return the standardised filename for a processed graph.

        Parameters
        ----------
        idx:
            Row index of the sample in the raw DataFrame.
        """
        return f"{self.name}_{idx}_{self.target_col}.pt"

    def _graph_filepath(self, idx: int) -> str:
        """Return the full path to the processed graph file for ``idx``."""
        return os.path.join(self.processed_dir, self._graph_filename(idx))

    def _one_hot_encode(
        self, x, allowable_set: list, allow_low_frequency: bool = False
    ) -> list[int]:
        """
        One-hot encode a value against an allowable set.

        Parameters
        ----------
        x:
            The value to encode.
        allowable_set:
            List of values that form the encoding vocabulary.
        allow_low_frequency:
            If True, append an extra ``1`` when ``x`` is not in
            ``allowable_set`` (wildcard / unknown token). If False,
            out-of-vocabulary values produce an all-zero vector.

        Returns
        -------
        list[int]
            One-hot encoded list, length ``len(allowable_set)`` or
            ``len(allowable_set) + 1`` when ``allow_low_frequency=True``.
        """
        vector = [int(x == s) for s in allowable_set]
        if allow_low_frequency:
            vector.append(int(x not in allowable_set))
        return vector

    def log_dataset_info(self) -> None:
        """Log a summary of the dataset size."""
        logger.info(f"{self.name} dataset: {len(self)} samples.")

    @property
    def _element_list(self) -> list[str]:
        """
        Common elements found in polymer SMILES strings.

        Used as a default allowable set for atomic number one-hot encoding.
        """
        return [
            "H",  # 1
            "Li",  # 3
            "B",  # 5
            "C",  # 6
            "N",  # 7
            "O",  # 8
            "F",  # 9
            "Na",  # 11
            "Si",  # 14
            "P",  # 15
            "S",  # 16
            "Cl",  # 17
            "K",  # 19
            "Se",  # 34
            "Br",  # 35
            "I",  # 53
        ]
