"""GUI rerun warnings and scoped removal of generated experiment outputs.

Core/CLI computations keep their cache behaviour. GUI stages call cleanup only
when their Run button is pressed, after the warning has been acknowledged.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import shutil
import tempfile

import pandas as pd
import streamlit as st

from polynet.app.services.cache import file_version, read_csv, read_json
from polynet.explainability.shap_explain import _class_key, _parse_model_log_name

logger = logging.getLogger(__name__)

_TRAINING_OUTPUTS = (
    "ml_results",
    "gnn_hyp_opt",
    "tml_hyp_opt",
    "split_indices.json",
    "split_options.json",
    "train_gnn_options.json",
    "train_tml_options.json",
    "preprocessing_tml_options.json",
    "target_transform_options.json",
    "hpo_search_spaces.json",
    "explanations",
    "unseen_predictions",
    "parity_plot.png",
)


@dataclass(frozen=True)
class RerunPlan:
    """Existing files affected by one GUI operation; inspecting it never writes."""

    paths: tuple[Path, ...]
    message: str
    cache_paths: tuple[Path, ...] = ()
    selection: tuple | None = None

    @property
    def has_results(self) -> bool:
        return bool(self.paths or self.cache_paths)

    @property
    def fingerprint(self) -> str:
        versions = [
            (str(path), path.lstat().st_mtime_ns, path.lstat().st_ctime_ns)
            for path in (*self.paths, *self.cache_paths)
        ]
        return hashlib.sha256(repr((versions, self.selection)).encode()).hexdigest()[:16]


def stage_rerun_plan(experiment_path: Path, stage: str) -> RerunPlan:
    """Representation invalidates all dependants; training preserves representation."""
    if stage not in {"representation", "training"}:
        raise ValueError(f"Unknown rerun stage: {stage}")
    names = _TRAINING_OUTPUTS
    if stage == "representation":
        names = ("representation", "representation_options.json", *names)
        message = (
            "Rebuilding representations will delete the existing representations, "
            "trained models and scalers, tuning results, predictions, metrics, plots, "
            "and explanations, including external prediction outputs. You will need to train again."
        )
    else:
        message = (
            "Rerunning training will delete the existing trained models and scalers, "
            "tuning results, predictions, metrics, plots, and explanations, including "
            "external prediction outputs. The saved representations will be reused."
        )
    root = Path(experiment_path)
    return RerunPlan(
        tuple(
            root / name for name in names if (root / name).exists() or (root / name).is_symlink()
        ),
        message,
    )


def rerun_warning(plan: RerunPlan, key: str) -> bool:
    """Warn before Run; new results or a changed selection require a fresh acknowledgement."""
    if not plan.has_results:
        return True
    st.warning(plan.message)
    return st.checkbox(
        "I understand these existing results will be replaced",
        value=False,
        key=f"rerun_ack:{key}:{plan.fingerprint}",
    )


def _forget_loaded_analysis() -> None:
    for key in list(st.session_state):
        if str(key).startswith("loaded_explanation_"):
            del st.session_state[key]


def clear_stage_results(experiment_path: Path, stage: str) -> None:
    """Remove only known generated outputs; keep source data and experiment settings."""
    plan = stage_rerun_plan(experiment_path, stage)
    if plan.has_results:
        logger.info(
            "Clearing %s outputs and downstream results (%s entries)", stage, len(plan.paths)
        )
    for path in plan.paths:
        if path.is_symlink() or not path.is_dir():
            path.unlink(missing_ok=True)
        else:
            shutil.rmtree(path)
    _forget_loaded_analysis()


def _selection(family: str, params: dict) -> tuple:
    models = sorted(params.get("model_names", params.get("models", {})))
    ids = {
        str(s) for s in params.get("explain_mols" if family == "GNN" else "explain_sample_ids", [])
    }
    restrictions = params.get("mols_per_model" if family == "GNN" else "samples_per_model")
    pairs = tuple(
        sorted(
            (name, sid)
            for name in models
            for sid in (
                ids if restrictions is None else ids & {str(s) for s in restrictions.get(name, [])}
            )
        )
    )
    class_key = (
        ("regression" if params.get("target_class") is None else str(params["target_class"]))
        if family == "GNN"
        else _class_key(params["problem_type"], params.get("target_class"))
    )
    fragmentation = str(params.get("fragmentation_approach")) if family == "GNN" else None
    return family, class_key, fragmentation, tuple(models), pairs


def _tml_mask(frame: pd.DataFrame, selection: tuple, descriptor: str) -> pd.Series:
    _, class_key, _, _, pairs = selection
    selected_pairs = {
        (_parse_model_log_name(name)[0], str(_parse_model_log_name(name)[2]), sid)
        for name, sid in pairs
        if _parse_model_log_name(name)[1] == descriptor
    }
    matches = [
        (str(row.model_type), str(row.iteration), str(row.sample_id)) in selected_pairs
        and str(row.class_idx) == class_key
        for row in frame[["model_type", "iteration", "sample_id", "class_idx"]].itertuples(
            index=False
        )
    ]
    return pd.Series(matches, index=frame.index, dtype=bool)


def _gnn_entries(raw: dict, selection: tuple):
    _, class_key, fragmentation, _, pairs = selection
    for name, sid in pairs:
        model, iteration = name.rsplit("_", 1)
        entry = raw.get(model, {}).get(iteration, {}).get(sid, {})
        monomers = entry.get("chemistry_masking", {}).get(class_key, {})
        for fragments in monomers.values():
            if fragmentation in fragments:
                yield fragments, fragmentation


@st.cache_data(show_spinner=False, max_entries=64)
def _snapshot_selection(version):
    # Cache only selection metadata, not a second copy of all raw scores.
    with open(version[0]) as file:
        record = json.load(file)
    family = record.get("family")
    return _selection(family, record["arguments"]) if family in {"TML", "GNN"} else None


@st.cache_data(show_spinner=False, max_entries=64)
def _gnn_has_scores(version, selection):
    with open(version[0]) as file:
        raw = json.load(file)
    return any(_gnn_entries(raw, selection))


def explanation_rerun_plan(family: str, params: dict) -> RerunPlan:
    """Identify only the chosen analysis and its model/sample/class/fragment scores."""
    root = Path(params.get("cache_root") or params["experiment_path"])
    selection = _selection(family, params)
    snapshots = []
    for path in (root / "explanations" / "analyses").glob("*.json"):
        if _snapshot_selection(file_version(path)) == selection:
            snapshots.append(path)
    caches = []
    if family == "TML":
        descriptors = sorted({_parse_model_log_name(name)[1] for name, _ in selection[4]})
        for descriptor in descriptors:
            path = root / "explanations" / f"shap_{descriptor}.csv"
            if path.exists():
                metadata = read_csv(
                    path, usecols=["model_type", "iteration", "sample_id", "class_idx"], dtype=str
                )
                if _tml_mask(metadata, selection, descriptor).any():
                    caches.append(path)
    else:
        path = root / "explanations" / "explanation.json"
        if path.exists() and _gnn_has_scores(file_version(path), selection):
            caches.append(path)
    return RerunPlan(
        tuple(snapshots),
        "Rerunning this explanation will replace the saved analysis for this selection "
        "and recalculate its model/bootstrap, molecule, class and fragmentation scores. "
        "Other saved analyses and predictions will be kept.",
        tuple(caches),
        selection,
    )


def explanation_rerun_button(label: str, family: str, params: dict, key: str) -> bool:
    """Keep the warning beside each page 5 Run button, before any deletion."""
    try:
        plan = explanation_rerun_plan(family, params)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Could not inspect previous explanation results: {exc}")
        return False
    allowed = rerun_warning(plan, key)
    return st.button(label, disabled=not allowed)


def _replace_file(path: Path, write) -> None:
    """Prepare a cache rewrite completely before replacing the previous file."""
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix=".tmp") as handle:
        temporary = Path(handle.name)
    try:
        write(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def clear_explanation_results(family: str, params: dict) -> None:
    """Forget selected raw scores before computation; preserve unrelated scores/snapshots."""
    plan = explanation_rerun_plan(family, params)
    if plan.has_results:
        logger.info(
            "Replacing selected %s explanation: %s model/sample pairs, %s saved analyses",
            family,
            len(plan.selection[4]),
            len(plan.paths),
        )
    for path in plan.cache_paths:
        if family == "TML":
            descriptor = path.stem.removeprefix("shap_")
            raw = read_csv(
                path,
                dtype={name: str for name in ["model_type", "iteration", "sample_id", "class_idx"]},
                float_precision="round_trip",
            )
            retained = raw.loc[~_tml_mask(raw, plan.selection, descriptor)]
            _replace_file(path, lambda temporary: retained.to_csv(temporary, index=False))
        else:
            raw = read_json(path)
            for fragments, key in _gnn_entries(raw, plan.selection):
                del fragments[key]
            _replace_file(path, lambda temporary: temporary.write_text(json.dumps(raw)))
    for path in plan.paths:
        path.unlink(missing_ok=True)
    _forget_loaded_analysis()
