"""Portable snapshots of completed GUI explanations, for page 6 plotting.

Snapshots contain selected raw scores and plotting inputs, never fitted models.
They preserve the population of each split and survive cache changes. Existing
CLI/GUI caches can also be opened without calculating missing attributions.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st

from polynet.app.services.cache import file_version, read_csv, read_json
from polynet.config.enums import FragmentationMethod, IteratorType, ProblemType
from polynet.config.paths import explanation_json_file_path, representation_file_path
from polynet.explainability.explain import build_display_data
from polynet.explainability.masking import MASKING_CACHE_VERSION, load_masking_cache
from polynet.explainability.shap_explain import (
    SHAP_CACHE_VERSION,
    _class_key,
    _load_shap_cache,
    _parse_model_log_name,
    _rows_of_models,
)

SNAPSHOT_VERSION = 1


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value, key=str)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot save explanation value of type {type(value).__name__}")


def _clean_metadata(value):
    if isinstance(value, dict):
        return {str(k): _clean_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean_metadata(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _pack_frame(frame):
    # Python's JSON float representation round-trips all significant digits.
    return _clean_metadata(
        {
            "columns": list(frame.columns),
            "index": frame.index.tolist(),
            "data": frame.to_numpy(dtype=object).tolist(),
        }
    )


def _unpack_frame(data):
    return pd.DataFrame(data["data"], index=data["index"], columns=data["columns"])


def capture_analysis(family: str, scope: str, params: dict) -> Path:
    """Snapshot a successful calculation's raw data and exact selection atomically."""
    root = Path(params.get("cache_root") or params["experiment_path"])
    models = dict.fromkeys(params["models"])
    ids = params["explain_mols" if family == "GNN" else "explain_sample_ids"]
    per_model = params.get("mols_per_model" if family == "GNN" else "samples_per_model")
    arguments = {
        "model_names": list(models),
        "problem_type": str(params["problem_type"]),
        "target_class": params.get("target_class"),
    }
    record = {
        "version": SNAPSHOT_VERSION,
        "attribution_version": MASKING_CACHE_VERSION if family == "GNN" else SHAP_CACHE_VERSION,
        "family": family,
        "scope": scope,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "arguments": arguments,
        "prediction_breakdowns": params.get("prediction_breakdowns") or {},
    }
    if family == "GNN":
        raw = load_masking_cache(read_json(explanation_json_file_path(root)))
        record["raw"] = build_display_data(raw, models, ids, per_model)
        arguments.update(
            explain_mols=ids,
            fragmentation_approach=str(params["fragmentation_approach"]),
            mols_per_model=per_model,
            predictions=params.get("predictions") or {},
        )
        selected = {str(s) for s in ids}
        record["molecules"] = [
            {"idx": mol.idx, "mols": list(mol.mols)}
            for mol in params["dataset"]
            if str(mol.idx) in selected
        ]
    else:
        arguments.update(
            explain_sample_ids=[str(s) for s in ids],
            samples_per_model=per_model,
            target_col=params.get("target_col"),
            predictions=params.get("predictions") or {},
        )
        record["raw"] = {}
        record["descriptors"] = {}
        descriptors = {str(k): v for k, v in params["descriptor_dfs"].items()}
        for descriptor in sorted({_parse_model_log_name(k)[1] for k in models}):
            frame = _load_shap_cache(root, descriptor)
            if frame.empty:
                continue
            frame = frame.loc[
                frame["class_idx"] == _class_key(params["problem_type"], params.get("target_class"))
            ]
            names = [k for k in models if _parse_model_log_name(k)[1] == descriptor]
            frame = _rows_of_models(frame, names, ids, per_model)
            record["raw"][descriptor] = _pack_frame(frame)
            values = descriptors.get(descriptor)
            if values is not None:
                values = values.copy()
                values.index = values.index.astype(str)
                values = values.loc[values.index.isin({str(s) for s in ids})]
                values = values.loc[:, values.columns.isin(frame.columns)]
                record["descriptors"][descriptor] = _pack_frame(values)
    directory = root / "explanations" / "analyses"
    directory.mkdir(parents=True, exist_ok=True)
    identity = json.dumps([family, scope, arguments], sort_keys=True, default=_json_default)
    key = hashlib.sha256(identity.encode()).hexdigest()[:20]
    destination = directory / f"{family.lower()}_{scope}_{key}.json"
    # Replace only after serialization succeeds, leaving any older snapshot intact.
    contents = json.dumps(_clean_metadata(record), default=_json_default, allow_nan=False)
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False, suffix=".tmp") as file:
        temporary = Path(file.name)
        file.write(contents)
    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def list_saved_analyses(root: Path) -> list[Path]:
    """Newest completed GUI runs first; temporary/incomplete files are excluded."""
    directory = Path(root) / "explanations" / "analyses"
    return sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)


def load_analysis(path: Path) -> dict:
    """Read a snapshot into rendering arguments; this never loads fitted models."""
    record = read_json(Path(path))
    if record.get("version") != SNAPSHOT_VERSION:
        raise ValueError(
            "Unsupported explanation snapshot version. Run the analysis again on page 5."
        )
    if record.get("family") not in {"GNN", "TML"} or record.get("scope") not in {"global", "local"}:
        raise ValueError("Invalid explanation snapshot family or scope.")
    expected = MASKING_CACHE_VERSION if record["family"] == "GNN" else SHAP_CACHE_VERSION
    if record.get("attribution_version") != expected:
        raise ValueError(
            "Attributions were calculated by an older version. Run the analysis again on page 5."
        )
    return _restore_record(record)


def _restore_record(record):
    arguments = record["arguments"].copy()
    arguments["problem_type"] = ProblemType(arguments["problem_type"])
    if record["family"] == "GNN":
        arguments["fragmentation_approach"] = FragmentationMethod(
            arguments["fragmentation_approach"]
        )
        arguments["combined_explanations"] = record["raw"]
        arguments["dataset"] = [SimpleNamespace(**mol) for mol in record["molecules"]]
    else:
        arguments["cache_by_descriptor"] = {k: _unpack_frame(v) for k, v in record["raw"].items()}
        arguments["descriptor_dfs"] = {
            k: _unpack_frame(v) for k, v in record["descriptors"].items()
        }
    return {**record, "arguments": arguments}


def list_cached_analyses(root: Path) -> list[dict]:
    """Describe existing compatible caches, including analyses run before snapshots.

    Cache-only imports have no record of the original set selection. They expose
    all cached samples explicitly; new GUI snapshots retain the exact population.
    """
    paths = sorted((Path(root) / "explanations").glob("shap_*.csv"))
    masking_path = explanation_json_file_path(root)
    if masking_path.exists():
        paths.append(masking_path)
    versions = tuple(file_version(p) for p in paths)
    return [
        {**entry, "cache_version": versions}
        for entry in _scan_caches(str(Path(root).resolve()), versions)
    ]


@st.cache_data(show_spinner=False, max_entries=8)
def _scan_caches(root: str, versions: tuple) -> list[dict]:
    # File fingerprints invalidate after CLI runs or atomic replacements.
    root = Path(root)
    entries = []
    by_class = {}
    for path in sorted((Path(root) / "explanations").glob("shap_*.csv")):
        descriptor = path.stem.removeprefix("shap_")
        frame = _load_shap_cache(root, descriptor)
        if frame.empty:
            continue
        for class_key, rows in frame.groupby("class_idx"):
            by_class.setdefault(str(class_key), {})[descriptor] = rows
    for class_key, frames in by_class.items():
        entries.append({"family": "TML", "class_key": class_key, "raw": frames})
    path = explanation_json_file_path(root)
    if path.exists():
        raw = load_masking_cache(read_json(path))
        groups = {}
        for model, iterations in raw.items():
            if not isinstance(iterations, dict):
                continue
            for iteration, samples in iterations.items():
                for sid, sample in samples.items():
                    for class_key, monomers in sample.get("chemistry_masking", {}).items():
                        for smiles, fragments in monomers.items():
                            for fragmentation, scores in fragments.items():
                                group = groups.setdefault((class_key, fragmentation), {})
                                entry = (
                                    group.setdefault(model, {})
                                    .setdefault(iteration, {})
                                    .setdefault(sid, {})
                                )
                                entry.setdefault("chemistry_masking", {}).setdefault(
                                    class_key, {}
                                ).setdefault(smiles, {})[fragmentation] = scores
        for (class_key, fragmentation), data in groups.items():
            entries.append(
                {
                    "family": "GNN",
                    "class_key": class_key,
                    "fragmentation": fragmentation,
                    "raw": data,
                }
            )
    return entries


def load_cached_analysis(entry: dict, root: Path, scope: str, data_options) -> dict:
    """Prepare a legacy cache for plotting, without inference or graph processing."""
    arguments = {
        "problem_type": (
            ProblemType.Regression
            if entry["class_key"] == "regression"
            else ProblemType.Classification
        ),
        "target_class": None if entry["class_key"] == "regression" else int(entry["class_key"]),
    }
    record = {**entry, "scope": scope, "arguments": arguments, "prediction_breakdowns": {}}
    if entry["family"] == "TML":
        frames = entry["raw"]
        arguments["model_names"] = sorted(
            {
                f"{row.model_type}-{descriptor}_{row.iteration}"
                for descriptor, frame in frames.items()
                for row in frame.itertuples()
            }
        )
        arguments["explain_sample_ids"] = sorted(
            {str(s) for frame in frames.values() for s in frame["sample_id"]}
        )
        arguments.update(cache_by_descriptor=frames, target_col=data_options.target_variable_col)
        descriptors = {}
        for descriptor in frames:
            path = representation_file_path(root) / f"{descriptor}.csv"
            if path.exists():
                descriptors[descriptor] = read_csv(path, index_col=0)
                descriptors[descriptor].index = descriptors[descriptor].index.astype(str)
        arguments["descriptor_dfs"] = descriptors
    else:
        raw = entry["raw"]
        arguments["model_names"] = sorted(
            f"{model}_{iteration}" for model, iterations in raw.items() for iteration in iterations
        )
        molecules = {}
        for iterations in raw.values():
            for samples in iterations.values():
                for sid, sample in samples.items():
                    molecules.setdefault(sid, list(sample["chemistry_masking"][entry["class_key"]]))
        arguments.update(
            combined_explanations=raw,
            explain_mols=list(molecules),
            dataset=[SimpleNamespace(idx=sid, mols=smiles) for sid, smiles in molecules.items()],
            fragmentation_approach=FragmentationMethod(entry["fragmentation"]),
        )
    predictions_path = root / "predictions.csv"
    if not predictions_path.exists():
        predictions_path = root / "ml_results" / "predictions.csv"
    if predictions_path.exists():
        predictions = read_csv(predictions_path, index_col=0)
        predictions.index = predictions.index.astype(str)
        iterator_cols = [c for c in predictions if c in {it.value for it in IteratorType}]
        iterator_col = iterator_cols[0] if iterator_cols else None
        if entry["family"] == "TML":
            from polynet.app.components.forms.explain_tml import _build_prediction_breakdowns

            record["prediction_breakdowns"] = _build_prediction_breakdowns(
                predictions,
                arguments["model_names"],
                arguments["explain_sample_ids"],
                data_options,
                iterator_col=iterator_col,
                selected_iterations=None,
            )
        else:
            from polynet.app.components.forms.explain_model import _build_gnn_prediction_breakdowns

            record["prediction_breakdowns"] = _build_gnn_prediction_breakdowns(
                predictions,
                arguments["model_names"],
                arguments["explain_mols"],
                data_options,
                iterator_col=iterator_col,
            )
    return record
