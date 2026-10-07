"""
tests/test_split_samplers.py
============================
astartes samplers for the train / validation / test splits. Fingerprint
samplers place each polymer by a *sampling fingerprint* (ratio-weighted
monomer count fingerprints) that is used for splitting only.
"""

import logging
from pathlib import Path
import warnings

from astartes import train_val_test_split
import numpy as np
import pandas as pd
from pydantic import ValidationError
import pytest

from polynet.config.enums import MolecularDescriptor, SplitSampler
from polynet.config.schemas import SplitConfig
from polynet.config.schemas.fingerprints import SamplingFingerprintConfig
from polynet.data.sampling import sampling_features, sampling_metadata
from polynet.factories.dataloader import get_data_split_indices
from polynet.featurizer.descriptors import (
    build_vector_representation,
    polymer_count_fingerprints,
    polymer_polybert_fingerprints,
)

_DATA = pd.read_csv(Path(__file__).parent / "fixtures" / "input_polymers.csv", index_col=0)
_DATA.index = [f"p{i}" for i in _DATA.index]
_SMILES = ["smiles_A", "smiles_B"]
_WEIGHTS = {"smiles_A": "weight_A", "smiles_B": "weight_B"}
_DATA["active"] = (_DATA["LogF_SA"] > _DATA["LogF_SA"].median()).astype(int)


def _split_cfg(**kwargs):
    return SplitConfig(split_type="train_val_test", test_ratio=0.2, val_ratio=0.2, **kwargs)


def _split(cfg, target="LogF_SA", n_iter=2):
    logging.disable(logging.WARNING)
    try:
        return get_data_split_indices(
            data=_DATA,
            split_type="train_val_test",
            n_bootstrap_iterations=n_iter,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
            target_variable_col=target,
            split_method=cfg.split_method,
            train_set_balance=None,
            random_seed=0,
            sampler=cfg.sampler,
            sampling_features=sampling_features(_DATA, _SMILES, _WEIGHTS, cfg),
        )
    finally:
        logging.disable(logging.NOTSET)


# --- configuration ---------------------------------------------------------------


def test_fingerprint_samplers_get_the_default_sampling_fingerprint():
    cfg = _split_cfg(sampler="kmeans")
    assert cfg.sampling_fingerprint.fingerprint == MolecularDescriptor.Morgan
    assert (cfg.sampling_fingerprint.fp_size, cfg.sampling_fingerprint.radius) == (2048, 3)
    assert _split_cfg().sampling_fingerprint is None  # random needs none


def test_sampling_fingerprint_is_ignored_with_a_warning_when_unused():
    with pytest.warns(UserWarning, match="has no effect"):
        cfg = _split_cfg(sampler="target_property", sampling_fingerprint={"fp_size": 64})
    assert cfg.sampling_fingerprint is None


def test_deterministic_sampler_with_repetitions_warns_but_is_allowed():
    with pytest.warns(UserWarning, match="deterministic"):
        cfg = _split_cfg(sampler="kennard_stone", n_bootstrap_iterations=3)
    assert cfg.n_bootstrap_iterations == 3


@pytest.mark.parametrize(
    "settings",
    [
        {"fingerprint": "rdkitfp", "radius": 2},
        {"fingerprint": "polymetrix"},
        {"fp_size": 0},
        {"fingerprint": "polybert", "fp_size": 512},
        {"fingerprint": "polybert", "radius": 2},
    ],
)
def test_invalid_sampling_fingerprints_are_rejected(settings):
    with pytest.raises(ValidationError):
        _split_cfg(sampler="kmeans", sampling_fingerprint=settings)


def test_polybert_sampling_fingerprint_takes_no_settings():
    fp = _split_cfg(sampler="kmeans", sampling_fingerprint={"fingerprint": "polybert"}).sampling_fingerprint
    assert (fp.fp_size, fp.radius) == (None, None) and not fp.is_count_fingerprint


# --- sampling fingerprints --------------------------------------------------------


@pytest.mark.parametrize("fingerprint, extra", [("morgan", {"radius": 2}), ("rdkitfp", {})])
def test_sampling_fingerprint_matches_the_weighted_representation(fingerprint, extra):
    cfg = SamplingFingerprintConfig(fingerprint=fingerprint, fp_size=256, **extra)
    X = polymer_count_fingerprints(_DATA, _SMILES, _WEIGHTS, cfg.fingerprint, cfg.settings())
    rep = build_vector_representation(
        _DATA.reset_index(names="id"),
        {cfg.fingerprint: cfg.settings().model_dump()},
        _SMILES,
        "id",
        "LogF_SA",
        "weighted_average",
        weights_col=_WEIGHTS,
    )[cfg.fingerprint]
    expected = rep[[c for c in rep.columns if c.startswith(fingerprint)]].to_numpy(float)
    assert np.array_equal(X, expected)


def test_polybert_sampling_fingerprint_matches_the_weighted_representation():
    pytest.importorskip("sentence_transformers")
    data = _DATA.head(12)
    X = polymer_polybert_fingerprints(data, _SMILES, _WEIGHTS)
    rep = build_vector_representation(
        data.reset_index(names="id"),
        {MolecularDescriptor.PolyBERT: True},
        _SMILES,
        "id",
        "LogF_SA",
        "weighted_average",
        weights_col=_WEIGHTS,
    )[MolecularDescriptor.PolyBERT]
    expected = rep[[c for c in rep.columns if c.startswith("polyBERT")]].to_numpy(float)
    assert X.shape == (12, 600) and np.allclose(X, expected)


# --- splitting --------------------------------------------------------------------


@pytest.mark.parametrize("sampler", [s for s in SplitSampler])
def test_every_sampler_gives_disjoint_complete_splits(sampler):
    cfg = _split_cfg(sampler=sampler, n_bootstrap_iterations=1)
    train, val, test = _split(cfg, n_iter=1)
    sets = [set(train[0]), set(val[0]), set(test[0])]
    assert all(sets) and not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    # Cluster samplers keep clusters whole; the rest uses every sample.
    assert set().union(*sets) <= set(_DATA.index)


def test_kennard_stone_uses_the_sampling_fingerprints():
    cfg = _split_cfg(sampler="kennard_stone", sampling_fingerprint={"fp_size": 512})
    train, val, test = _split(cfg, n_iter=1)
    X = sampling_features(_DATA, _SMILES, _WEIGHTS, cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        *_, tr, va, te = train_val_test_split(
            X, train_size=0.6, val_size=0.2, test_size=0.2, sampler="kennard_stone",
            hopts={}, return_indices=True,
        )
    assert set(test[0]) == set(_DATA.index[te]) and set(train[0]) == set(_DATA.index[tr])


def test_changing_the_sampling_fingerprint_changes_the_split():
    a = _split(_split_cfg(sampler="kennard_stone", sampling_fingerprint={"radius": 1}), n_iter=1)
    b = _split(_split_cfg(sampler="kennard_stone", sampling_fingerprint={"radius": 3}), n_iter=1)
    assert set(a[2][0]) != set(b[2][0])


def test_stratified_fingerprint_sampling_keeps_class_proportions():
    cfg = _split_cfg(sampler="kennard_stone", split_method="stratified")
    train, val, test = _split(cfg, target="active", n_iter=1)
    for part in (train[0], val[0], test[0]):
        assert _DATA.loc[part, "active"].mean() == pytest.approx(0.5, abs=0.06)


def test_repeated_samples_from_a_sampler_are_rejected(monkeypatch):
    import polynet.factories.dataloader as dataloader

    def broken_sampler(X, *args, **kwargs):  # e.g. NaN distances: one sample repeated
        return X, X, X, np.zeros(6, dtype=int), np.arange(1, 3), np.arange(3, 5)

    monkeypatch.setattr(dataloader, "train_val_test_split", broken_sampler)
    with pytest.raises(ValueError, match="repeated or overlapping"):
        dataloader.astartes_split(10, 0.2, 0.2, 0)


@pytest.mark.parametrize("sampler", ["spxy", "target_property"])
def test_target_samplers_cannot_be_stratified(sampler):
    # Within a class the target is constant: SPXY's target distances become NaN
    # (astartes then repeats one sample), target_property has nothing to order.
    with pytest.raises(ValidationError, match="constant within a class"):
        _split_cfg(sampler=sampler, split_method="stratified")


def test_target_property_puts_extreme_targets_in_the_test_set():
    train, _, test = _split(_split_cfg(sampler="target_property"), n_iter=1)
    assert _DATA.loc[test[0], "LogF_SA"].min() >= _DATA.loc[train[0], "LogF_SA"].max()


# --- metadata ---------------------------------------------------------------------


def test_sampling_metadata_records_the_sampler_and_fingerprint():
    cfg = _split_cfg(sampler="spxy", sampling_fingerprint={"fingerprint": "rdkitfp", "fp_size": 1024})
    meta = sampling_metadata(cfg, _WEIGHTS)
    assert meta["sampler"] == "spxy" and meta["uses_target"] and meta["deterministic"]
    assert meta["sampling_fingerprint"]["fingerprint"] == "rdkitfp"
    assert meta["sampling_fingerprint"]["fp_size"] == 1024
    assert meta["sampling_fingerprint"]["weights_col"] == _WEIGHTS
    assert meta["astartes_version"]
    assert sampling_metadata(_split_cfg(), _WEIGHTS)["sampling_fingerprint"] is None
    polybert = _split_cfg(sampler="kmeans", sampling_fingerprint={"fingerprint": "polybert"})
    assert sampling_metadata(polybert, _WEIGHTS)["sampling_fingerprint"]["model"] == "xushijie/polyBERT"


# --- samplers offered in the GUI --------------------------------------------------


@pytest.mark.parametrize("problem_type", ["regression", "classification"])
@pytest.mark.parametrize("split_method", ["random", "stratified"])
def test_every_offered_sampler_gives_a_valid_config(problem_type, split_method):
    from polynet.config.schemas.split_data import available_samplers

    offered = available_samplers(problem_type, split_method)
    assert offered[0] == SplitSampler.Random
    for sampler in offered:
        _split_cfg(sampler=sampler, split_method=split_method)  # must not raise
    for sampler in set(SplitSampler) - set(offered):
        if split_method == "stratified" and sampler in {SplitSampler.SPXY, SplitSampler.TargetProperty}:
            with pytest.raises(ValidationError):
                _split_cfg(sampler=sampler, split_method=split_method)
        else:  # hidden because it is meaningless here, not invalid
            assert problem_type == "classification" and sampler == SplitSampler.TargetProperty
