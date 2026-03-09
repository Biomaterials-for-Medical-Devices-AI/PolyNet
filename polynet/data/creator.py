from __future__ import annotations

from typing import Callable

import pandas as pd
from polymetrix.datasets import CuratedGlassTempDataset

from polynet.config.enums import DatasetName


class DatasetCreator:
    """
    Factory for creating built-in benchmark datasets used in PolyNet.

    This class provides a uniform interface for loading curated datasets
    and applying any PolyNet-specific column selection or renaming needed
    before downstream processing.

    Parameters
    ----------
    dataset_name:
        Name of the dataset to create. May be provided as a
        ``DatasetName`` enum member or its string value.

    Examples
    --------
    >>> creator = DatasetCreator(DatasetName.CuratedTg)
    >>> df = creator.create_dataset()
    >>> df.columns.tolist()
    ['PSMILES', 'Tg(K)']
    """

    def __init__(self, dataset_name: DatasetName | str = DatasetName.CuratedTg) -> None:
        self.dataset_name = (
            DatasetName(dataset_name) if isinstance(dataset_name, str) else dataset_name
        )

    def create_dataset(self) -> pd.DataFrame:
        """
        Create and return the requested dataset.

        Returns
        -------
        pd.DataFrame
            Dataset formatted for use in PolyNet.

        Raises
        ------
        ValueError
            If the requested dataset is not supported.
        """
        dataset_loaders: dict[DatasetName, Callable[[], pd.DataFrame]] = {
            DatasetName.CuratedTg: self._load_curated_tg
        }

        try:
            return dataset_loaders[self.dataset_name]()
        except KeyError as exc:
            supported = [d.value for d in dataset_loaders]
            raise ValueError(
                f"Dataset '{self.dataset_name}' is not currently supported. "
                f"Supported datasets: {supported}."
            ) from exc

    @staticmethod
    def _load_curated_tg() -> pd.DataFrame:
        """
        Load the curated glass transition temperature dataset.

        Returns
        -------
        pd.DataFrame
            DataFrame containing:
            - ``PSMILES``: polymer structure string
            - ``Tg(K)``: glass transition temperature in Kelvin
        """
        dataset = CuratedGlassTempDataset().df.copy()
        dataset = dataset[["PSMILES", "labels.Exp_Tg(K)"]]
        dataset = dataset.rename(columns={"labels.Exp_Tg(K)": "Tg(K)"})
        dataset.index.name = "ID"
        dataset = dataset.reset_index()
        return dataset
