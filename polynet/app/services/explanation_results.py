"""Portable snapshots of completed GUI explanations, for page 6 plotting.

Snapshots contain selected raw scores and plotting inputs, never fitted models.
They preserve the population of each split and survive cache changes. Population and molecule views share the same saved model/sample pairs.
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

from polynet.app.services.cache import file_version, read_json
from polynet.config.constants import DataSet
from polynet.config.display_names import prettify_label
from polynet.config.enums import FragmentationMethod, ProblemType
from polynet.config.paths import explanation_json_file_path
from polynet.explainability.explain import build_display_data
from polynet.explainability.masking import MASKING_CACHE_VERSION, load_masking_cache
from polynet.explainability.shap_explain import (
    SHAP_CACHE_VERSION,
    _class_key,
    _load_shap_cache,
    _parse_model_log_name,
    _rows_of_models,
)

SNAPSHOT_VERSION = 2


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
        "selection_label": params.get("selection_label"),
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
    pairs = attribution_coverage(_restore_record(record))
    identity = json.dumps(
        [
            family,
            sorted(models),
            arguments.get("target_class"),
            arguments.get("fragmentation_approach"),
            sorted((sid, sorted(names)) for sid, names in pairs.items()),
        ],
        sort_keys=True,
        default=_json_default,
    )
    key = hashlib.sha256(identity.encode()).hexdigest()[:20]
    destination = directory / f"{family.lower()}_{key}.json"
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
    if record.get("version") not in {1, SNAPSHOT_VERSION}:
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


def attribution_coverage(record: dict) -> dict[str, set[str]]:
    """Return actual saved model/bootstrap combinations per molecule."""
    args = record["arguments"]
    coverage = {}
    if record["family"] == "TML":
        for descriptor, frame in args["cache_by_descriptor"].items():
            for row in frame.itertuples():
                name = f"{row.model_type}-{descriptor}_{row.iteration}"
                coverage.setdefault(str(row.sample_id), set()).add(name)
    else:
        class_key = "regression" if args.get("target_class") is None else str(args["target_class"])
        fragmentation = str(args["fragmentation_approach"])
        for model, iterations in args["combined_explanations"].items():
            for iteration, samples in iterations.items():
                for sid, sample in samples.items():
                    monomers = sample.get("chemistry_masking", {}).get(class_key, {})
                    if any(fragmentation in fragments for fragments in monomers.values()):
                        coverage.setdefault(str(sid), set()).add(f"{model}_{iteration}")
    return coverage


def attribution_set_membership(record: dict) -> dict[str, dict[str, str]]:
    """Saved split membership per actual model/bootstrap/sample attribution.

    Membership belongs to a fitted bootstrap, not to the molecule globally.
    External data and snapshots without membership do not invent train/test sets.
    """
    coverage = attribution_coverage(record)
    saved = record.get("prediction_breakdowns") or {}
    entries = saved.values() if record["family"] == "TML" else [saved]
    membership = {}
    for group in entries:
        for sid, entry in group.items():
            names = {
                (prettify_label(name.rsplit("_", 1)[0]), int(name.rsplit("_", 1)[1])): name
                for name in coverage.get(str(sid), ())
            }
            for row in entry["rows"]:
                if row.get("Set") not in {DataSet.Training, DataSet.Validation, DataSet.Test}:
                    continue
                if "Bootstrap" not in row:
                    continue
                name = names.get((row["Model"], int(row["Bootstrap"])))
                if name is not None:
                    membership.setdefault(str(sid), {})[name] = row["Set"]
    return membership


def select_attribution_coverage(
    record: dict, model_names: list[str], set_name: str | None = None
) -> dict[str, set[str]]:
    """Intersect saved pairs with selected models and each bootstrap's own set."""
    coverage = attribution_coverage(record)
    membership = attribution_set_membership(record) if set_name is not None else {}
    selected = set(model_names)
    result = {}
    for sid, names in coverage.items():
        matches = names & selected
        if set_name is not None:
            matches = {name for name in matches if membership.get(sid, {}).get(name) == set_name}
        if matches:
            result[sid] = matches
    return result


def analysis_summary(path: Path) -> dict:
    """Read a small cached description for the analysis selector."""
    return _summary(file_version(path))


@st.cache_data(show_spinner=False, max_entries=64)
def _summary(version) -> dict:
    with open(version[0]) as file:
        record = json.load(file)
    args = record["arguments"]
    return {
        "family": record["family"],
        "model_names": args["model_names"],
        "sample_ids": args.get("explain_mols", args.get("explain_sample_ids", [])),
        "target_class": args.get("target_class"),
        "fragmentation": args.get("fragmentation_approach"),
        "selection_label": record.get("selection_label"),
    }


def analysis_label(summary: dict) -> str:
    """Describe scientific selections, without run dates, hashes or view types."""
    models = sorted({prettify_label(name.rsplit("_", 1)[0]) for name in summary["model_names"]})
    splits = sorted({int(name.rsplit("_", 1)[1]) for name in summary["model_names"]})
    ids = summary["sample_ids"]
    selection = summary.get("selection_label")
    if not selection:
        selection = ", ".join(map(str, ids[:3])) + (", …" if len(ids) > 3 else "")
    parts = [
        summary["family"],
        ", ".join(models),
        f"splits {', '.join(map(str, splits))}",
        f"{selection} ({len(ids)} samples)",
    ]
    if summary.get("fragmentation"):
        parts.append(prettify_label(summary["fragmentation"]))
    if summary.get("target_class") is not None:
        parts.append(f"class {summary['target_class']}")
    return " · ".join(parts)


def local_prediction_breakdowns(
    record: dict,
    sample_ids: list,
    model_names: list[str] | None = None,
    samples_per_model: dict[str, set[str]] | None = None,
) -> dict:
    """Keep prediction rows for the exact saved model/sample pairs being plotted."""
    coverage = attribution_coverage(record)
    saved = record.get("prediction_breakdowns") or {}

    def subset(entries, descriptor=None):
        result = {}
        for sid in map(str, sample_ids):
            entry = entries.get(sid)
            if entry is None:
                continue
            allowed = {
                (prettify_label(name.rsplit("_", 1)[0]), int(name.rsplit("_", 1)[1]))
                for name in coverage.get(sid, ())
                if (model_names is None or name in model_names)
                and (samples_per_model is None or sid in samples_per_model.get(name, ()))
                and (descriptor is None or _parse_model_log_name(name)[1] == descriptor)
            }
            rows = [
                row
                for row in entry["rows"]
                if "Bootstrap" not in row or (row["Model"], int(row["Bootstrap"])) in allowed
            ]
            result[sid] = {**entry, "rows": rows}
        return result

    if record["family"] == "TML":
        return {descriptor: subset(entries, descriptor) for descriptor, entries in saved.items()}
    return subset(saved)
