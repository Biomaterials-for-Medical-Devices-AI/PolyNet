"""
polynet.applicability.assess
============================
Applicability domain of new polymers for the models of a trained experiment.

For every split, the new polymers are placed relative to the polymers that
split's models were trained on (``KNNApplicabilityDomain``), and the held-out
test polymers of the split are placed the same way to relate the distance to
the test error (``DistanceErrorCalibration``). The per-split results are
summarised over the splits, like the ensembles.

Each metric of ``ApplicabilityDomainConfig.metric`` gives its own domains:

- ``ruzicka_morgan`` — Ruzicka distance on the count fingerprint, one domain
  per model family (``{scope}`` = ``GNN`` / ``TML``);
- ``euclidean_model_inputs`` — Euclidean distance in the scaled,
  feature-selected inputs of the traditional models, one domain per
  representation (``{scope}`` = ``TML rdkit``, ``TML morgan``, …). GNNs have
  no such inputs and only use ``ruzicka_morgan``.

Columns added to the predictions (``{metric}`` is ``Ruzicka`` or
``Euclidean``, ``{name}`` an ensemble name such as ``GCN`` or
``random forest-rdkit``):

- ``{scope} AD {metric} Score`` — mean distance to the ``k`` nearest
  training polymers divided by the domain cutoff (1 = boundary, above 1 =
  outside), averaged over the splits;
- ``{scope} AD {metric} In Domain Fraction`` — share of the splits in whose
  domain the polymer lies;
- ``{scope} AD {metric} In Domain`` — in the domain of at least half of the
  splits;
- ``{family} AD Descriptors In Range Fraction`` — share of the splits whose
  training range contains the polymer descriptors (only with
  ``polymer_descriptors``; folded into every in-domain flag);
- ``{name} AD {metric} Expected Abs Error {target}`` (regression) or
  ``{name} AD {metric} Expected Accuracy {target}`` (classification) — mean
  held-out test error of the split models at a similar score, averaged over
  the splits; empty when the polymer is farther from the training set than
  every test polymer (no error was observed that far out).

::

    from polynet.applicability.assess import ModelGroup, assess_applicability_domain
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd

from polynet.applicability.calibration import DistanceErrorCalibration, prediction_errors
from polynet.applicability.domain import KNNApplicabilityDomain
from polynet.applicability.reference import (
    GNN,
    TML,
    load_feature_transformer,
    load_split_ids,
    load_training_dataset,
    load_training_descriptors,
    load_training_predictions,
    reference_ids,
    tml_trained_with_validation,
)
from polynet.config.column_names import (
    get_iterator_name,
    get_predicted_label_column_name,
    get_true_label_column_name,
)
from polynet.config.constants import DataSet, ResultColumn
from polynet.config.enums import ApplicabilityDomainMetric, ProblemType, SplitType
from polynet.config.schemas import ApplicabilityDomainConfig, DataConfig, RepresentationConfig

logger = logging.getLogger(__name__)

METRIC_LABELS = {
    ApplicabilityDomainMetric.RuzickaMorgan: "Ruzicka",
    ApplicabilityDomainMetric.EuclideanModelInputs: "Euclidean",
}
AD_DESCRIPTORS_IN_RANGE = "AD Descriptors In Range Fraction"
_IN_DOMAIN_COLUMN = re.compile(
    rf"^(?P<scope>.+) AD (?P<metric>{'|'.join(METRIC_LABELS.values())}) In Domain$"
)

METHOD = {
    "domain": "k-nearest-neighbour distance, cutoff <d> + Z·sigma; score = distance / cutoff "
    "(Tropsha, Gramatica & Gombar, QSAR Comb. Sci. 2003, 22, 69-77)",
    "ruzicka_morgan": "1 - min-max (Ruzicka) similarity of ratio-weighted count fingerprints",
    "euclidean_model_inputs": "Euclidean distance in the scaled, feature-selected inputs of "
    "the traditional models of each split",
    "expected_error": "held-out test error binned by domain score "
    "(Sheridan et al., J. Chem. Inf. Comput. Sci. 2004, 44, 1912-1928)",
}


@dataclass
class ModelGroup:
    """
    Models summarised together, as in the ensembles.

    Attributes
    ----------
    name:
        Ensemble name used in the column names (e.g. ``"GCN"``,
        ``"random forest-rdkit"``).
    family:
        ``"GNN"`` or ``"TML"``; sets the training reference.
    model_ids:
        Model names as they appear in the experiment's
        ``ml_results/predictions.csv`` (e.g. ``"GCN"``,
        ``"random_forest-rdkit"``), used to read their test errors.
    representation:
        Representation of TML models (e.g. ``"rdkit"``), for
        ``euclidean_model_inputs``; ``None`` for GNNs.
    """

    name: str
    family: str
    model_ids: list[str] = field(default_factory=list)
    representation: str | None = None


@dataclass
class _Space:
    """One domain: a metric over a set of models sharing a reference and inputs."""

    metric: ApplicabilityDomainMetric
    scope: str
    family: str
    groups: list[ModelGroup]
    # split → (training inputs indexed by sample ID, new-polymer inputs)
    inputs: Callable[[str], tuple[pd.DataFrame, np.ndarray]]


def assess_applicability_domain(
    data: pd.DataFrame,
    groups: list[ModelGroup],
    experiment_path: Path,
    data_cfg: DataConfig,
    repr_cfg: RepresentationConfig,
    ad_cfg: ApplicabilityDomainConfig,
    new_descriptors: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Applicability domain columns for new polymers, and a report of how they were made.

    Parameters
    ----------
    data:
        New polymers (prepared like the training data), one row per
        prediction, in the order of the predictions.
    groups:
        Model groups to report the expected error for.
    experiment_path:
        Root of the trained experiment.
    data_cfg, repr_cfg:
        Data and representation settings of the experiment.
    ad_cfg:
        Applicability domain settings.
    new_descriptors:
        TML descriptors of the new polymers by representation (cleaned like
        the training data, rows in the order of ``data``); needed for
        ``euclidean_model_inputs``.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        ``(columns, report)``: the columns described in the module docstring
        (positional index, one row per row of ``data``), and the settings,
        per-split domains and calibration bins.
    """
    training = load_training_dataset(experiment_path, data_cfg)
    splits = load_split_ids(experiment_path)
    if splits is None:
        logger.warning(
            "The experiment has no split_indices.json: the applicability domain uses the "
            "whole dataset as reference and the expected error cannot be estimated."
        )
        splits = {"all": {"train": list(training.index), "val": [], "test": []}}

    spaces = _spaces(
        data, groups, training, experiment_path, data_cfg, repr_cfg, ad_cfg, new_descriptors
    )

    descriptor_cols = repr_cfg.polymer_descriptors or []
    train_desc = training[descriptor_cols] if descriptor_cols else None
    new_desc = data[descriptor_cols].reset_index(drop=True) if descriptor_cols else None

    predictions = load_training_predictions(experiment_path)
    if predictions is None:
        logger.warning(
            "The experiment has no ml_results/predictions.csv: the expected error cannot be "
            "estimated."
        )

    columns: dict[str, np.ndarray] = {}
    report: dict = {
        "method": METHOD,
        "settings": ad_cfg.model_dump(mode="json"),
        "domains": {},
        "expected_error": {},
    }
    for space in spaces:
        label = METRIC_LABELS[space.metric]
        scores, in_domain, in_range = [], [], []
        pairs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {g.name: [] for g in space.groups}
        domains: dict[str, KNNApplicabilityDomain] = {}

        for split, ids in splits.items():
            train_inputs, new_inputs = space.inputs(split)
            ref = _rows(train_inputs, reference_ids(ids, space.family, experiment_path), split)
            domain = KNNApplicabilityDomain(
                k=ad_cfg.k_neighbours, z=ad_cfg.z, metric=space.metric
            ).fit(ref.to_numpy(), None if train_desc is None else train_desc.loc[ref.index])
            domains[split] = domain
            scored = domain.score(new_inputs, new_desc)
            scores.append(scored["score"].to_numpy())
            in_domain.append(scored["in_domain"].to_numpy())
            in_range.append(scored["descriptors_in_range"].to_numpy())

            if predictions is None or not ids["test"]:
                continue
            test = _rows(train_inputs, ids["test"], split)
            test_score = pd.Series(
                domain.score(
                    test.to_numpy(), None if train_desc is None else train_desc.loc[test.index]
                )["score"].to_numpy(),
                index=test.index,
            )
            for group in space.groups:
                errors = _test_errors(predictions, group, split, data_cfg)
                if errors is not None:
                    errors = errors[errors.index.isin(test_score.index)]
                    pairs[group.name].append(
                        (test_score.loc[errors.index].to_numpy(), errors.to_numpy())
                    )

        fraction = np.mean(in_domain, axis=0)
        columns[f"{space.scope} AD {label} Score"] = np.mean(scores, axis=0)
        columns[f"{space.scope} AD {label} In Domain Fraction"] = fraction
        columns[f"{space.scope} AD {label} In Domain"] = fraction >= 0.5
        if descriptor_cols:
            columns.setdefault(
                f"{space.family} {AD_DESCRIPTORS_IN_RANGE}", np.mean(in_range, axis=0)
            )
        report["domains"][f"{space.scope} ({space.metric.value})"] = {
            "reference": (
                "training + validation"
                if space.family == TML and tml_trained_with_validation(experiment_path)
                else "training"
            ),
            "splits": {split: d.summary() for split, d in domains.items()},
        }
        logger.info(
            f"Applicability domain ({space.scope}, {label}): {int((fraction >= 0.5).sum())} of "
            f"{len(fraction)} polymers in domain."
        )

        for group in space.groups:
            if not pairs[group.name]:
                continue
            calibration = DistanceErrorCalibration(n_bins=ad_cfg.n_bins).fit(
                np.concatenate([s for s, _ in pairs[group.name]]),
                np.concatenate([e for _, e in pairs[group.name]]),
            )
            # Mean over the splits that observed test errors this far out (NaN if none).
            columns[expected_error_column(group.name, space.metric, data_cfg)] = (
                pd.DataFrame([calibration.predict(s) for s in scores]).mean().to_numpy()
            )
            report["expected_error"][f"{group.name} ({space.metric.value})"] = calibration.summary()

    return pd.DataFrame(columns), report


def _spaces(
    data: pd.DataFrame,
    groups: list[ModelGroup],
    training: pd.DataFrame,
    experiment_path: Path,
    data_cfg: DataConfig,
    repr_cfg: RepresentationConfig,
    ad_cfg: ApplicabilityDomainConfig,
    new_descriptors: dict[str, pd.DataFrame] | None,
) -> list[_Space]:
    """The domains to assess, one per metric and scope."""
    spaces: list[_Space] = []

    if ApplicabilityDomainMetric.RuzickaMorgan in ad_cfg.metric:
        from polynet.featurizer.descriptors import polymer_count_fingerprints

        fp_cfg = ad_cfg.fingerprint

        def fingerprints(df: pd.DataFrame) -> np.ndarray:
            return polymer_count_fingerprints(
                df,
                data_cfg.smiles_cols,
                repr_cfg.weights_col,
                fp_cfg.fingerprint,
                fp_cfg.settings(),
            )

        train_fps = pd.DataFrame(fingerprints(training), index=training.index)
        new_fps = fingerprints(data.reset_index(drop=True))
        for family in dict.fromkeys(g.family for g in groups):
            spaces.append(
                _Space(
                    ApplicabilityDomainMetric.RuzickaMorgan,
                    family,
                    family,
                    [g for g in groups if g.family == family],
                    lambda split: (train_fps, new_fps),
                )
            )

    if ApplicabilityDomainMetric.EuclideanModelInputs in ad_cfg.metric:
        if any(g.family == GNN for g in groups):
            logger.info("GNNs have no tabular inputs; their domain uses ruzicka_morgan only.")
        weights_cols = list(repr_cfg.weights_col.values()) if repr_cfg.weights_col else None
        tml_groups = [g for g in groups if g.family == TML and g.representation]
        for rep in dict.fromkeys(g.representation for g in tml_groups):
            train_desc = load_training_descriptors(experiment_path, rep, data_cfg, weights_cols)
            new_desc = (new_descriptors or {}).get(rep)
            if new_desc is not None:
                new_desc = new_desc.drop(columns=[data_cfg.target_variable_col], errors="ignore")
            if train_desc is None or new_desc is None:
                logger.warning(
                    f"No '{rep}' descriptors of the training or new polymers: the "
                    "euclidean_model_inputs domain of this representation is skipped."
                )
                continue
            spaces.append(
                _Space(
                    ApplicabilityDomainMetric.EuclideanModelInputs,
                    f"{TML} {rep}",
                    TML,
                    [g for g in tml_groups if g.representation == rep],
                    _model_inputs(
                        experiment_path, rep, train_desc, new_desc.reset_index(drop=True)
                    ),
                )
            )
    return spaces


def _model_inputs(
    experiment_path: Path, rep: str, train_desc: pd.DataFrame, new_desc: pd.DataFrame
) -> Callable[[str], tuple[pd.DataFrame, np.ndarray]]:
    """Inputs of a representation as each split's models see them (scaled, selected)."""

    def inputs(split: str) -> tuple[pd.DataFrame, np.ndarray]:
        transformer = load_feature_transformer(experiment_path, rep, split)
        if transformer is None:
            train, new = train_desc.to_numpy(dtype=float), new_desc.to_numpy(dtype=float)
        else:
            train, new = transformer.transform(train_desc), transformer.transform(new_desc)
        # Undefined descriptors are imputed with the training mean, as in prediction.
        return (
            pd.DataFrame(np.nan_to_num(np.asarray(train, dtype=float)), index=train_desc.index),
            np.nan_to_num(np.asarray(new, dtype=float)),
        )

    return inputs


def summarise_domain(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Per domain (scope and metric), how many predicted polymers are in it.

    Parameters
    ----------
    predictions:
        Predictions with the columns of ``assess_applicability_domain``.

    Returns
    -------
    pd.DataFrame
        One row per domain (index, e.g. ``"TML · Ruzicka"``) with
        ``In domain``, ``Out of domain`` and ``Mean score``; empty when the
        predictions have no domain columns.
    """
    rows = {}
    for col in predictions.columns:
        match = _IN_DOMAIN_COLUMN.match(col)
        if match is None:
            continue
        scope, metric = match["scope"], match["metric"]
        in_domain = predictions[col].astype(bool)
        rows[f"{scope} · {metric}"] = {
            "In domain": int(in_domain.sum()),
            "Out of domain": int((~in_domain).sum()),
            "Mean score": float(predictions[f"{scope} AD {metric} Score"].mean()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def expected_error_column(
    name: str, metric: ApplicabilityDomainMetric | str, data_cfg: DataConfig
) -> str:
    """``{name} AD {metric} Expected Abs Error {target}`` (or ``Expected Accuracy``)."""
    error = (
        "Expected Accuracy"
        if data_cfg.problem_type == ProblemType.Classification
        else "Expected Abs Error"
    )
    suffix = f" {data_cfg.target_variable_name}" if data_cfg.target_variable_name else ""
    return f"{name} AD {METRIC_LABELS[ApplicabilityDomainMetric(metric)]} {error}{suffix}"


def _rows(inputs: pd.DataFrame, ids: list[str], split: str) -> pd.DataFrame:
    """Inputs of the given sample IDs; the IDs must be in the training dataset."""
    missing = pd.Index(ids).difference(inputs.index)
    if len(missing):
        raise ValueError(
            f"Split {split}: {len(missing)} sample ID(s) of split_indices.json are not in the "
            f"training dataset, e.g. {list(missing[:5])}."
        )
    return inputs.loc[ids]


def _test_errors(
    predictions: pd.DataFrame, group: ModelGroup, split: str, data_cfg: DataConfig
) -> pd.Series | None:
    """Held-out test errors of a group's models in one split, indexed by sample ID."""
    iterator = get_iterator_name(SplitType.TrainValTest)
    test = predictions[
        (predictions[ResultColumn.SET] == DataSet.Test)
        & (predictions[iterator].astype(str) == split)
    ].set_index(ResultColumn.INDEX)
    # TML columns use the target column name when no display name is set.
    names = [
        data_cfg.target_variable_name,
        data_cfg.target_variable_name or data_cfg.target_variable_col,
    ]
    label = next((c for c in map(get_true_label_column_name, names) if c in test.columns), None)
    errors = []
    for model_id in group.model_ids:
        pred = next(
            (
                c
                for c in (get_predicted_label_column_name(n, model_id) for n in names)
                if c in test.columns
            ),
            None,
        )
        if label is None or pred is None:
            logger.warning(
                f"No test predictions of '{model_id}' in ml_results/predictions.csv; it is "
                "left out of the expected error."
            )
            continue
        rows = test[[label, pred]].dropna()
        errors.append(prediction_errors(rows[label], rows[pred], data_cfg.problem_type))
    return pd.concat(errors) if errors else None
