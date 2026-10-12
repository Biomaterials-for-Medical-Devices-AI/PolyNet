from __future__ import annotations

from importlib.resources import files
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
            DatasetName.CuratedTg: self._load_curated_tg,
            DatasetName.FluorineNMRSNR: self._load_fluorine_nmr_snr,
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

    @staticmethod
    def _load_fluorine_nmr_snr() -> pd.DataFrame:
        """
        Load the 19F NMR signal-to-noise ratio dataset of fluorinated copolymers.

        Bundled with PolyNet (``polynet/data/benchmarks/fluorine_nmr_snr.csv``).
        Every polymer is a copolymer of the same six acrylate monomers; only
        their molar ratios change.

        References: Reis, M. et al. Machine-Learning-Guided Discovery of 19F
        MRI Agents Enabled by Automated Copolymer Synthesis. J. Am. Chem. Soc.
        143, 17677–17689 (2021); Tao, L., Arbaugh, T., Byrnes, J., Varshney, V.
        & Li, Y. Unified machine learning protocol for copolymer
        structure-property predictions. STAR Protocols 3 (2022).

        Returns
        -------
        pd.DataFrame
            DataFrame containing:
            - ``ID``: polymer identifier
            - ``smiles_1`` … ``smiles_6``: monomer PSMILES
            - ``ratio_1`` … ``ratio_6``: molar ratios (each row sums to 1)
            - ``SNR``: 19F NMR signal-to-noise ratio
            - ``MolWt``, ``Dispersity``: measured molecular weight and
              dispersity, missing (NaN) for most polymers
        """
        with files("polynet.data").joinpath("benchmarks", "fluorine_nmr_snr.csv").open() as f:
            return pd.read_csv(f)


def subsample_dataset(df: pd.DataFrame, n_samples: int, seed: int = 0) -> pd.DataFrame:
    """
    A reproducible random subset of ``n_samples`` rows, in their original order.

    The rows are shuffled once with ``seed`` and the first ``n_samples`` are
    kept, so for the same seed a smaller subset is always contained in a
    larger one.

    Parameters
    ----------
    df:
        The dataset.
    n_samples:
        Number of rows to keep; the whole dataset when at least ``len(df)``.
    seed:
        Seed of the shuffle.

    Returns
    -------
    pd.DataFrame
        The selected rows, with their original index.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}.")
    if n_samples >= len(df):
        return df
    return df.sample(frac=1.0, random_state=seed).head(n_samples).sort_index()
