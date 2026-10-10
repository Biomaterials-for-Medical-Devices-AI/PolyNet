"""Transparent workload guidance, not a hardware-dependent runtime prediction."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Workload:
    final_fits: int
    fit_budget: int
    gnn_epochs: int
    reasons: tuple[str, ...]


def estimate_workload(
    *,
    rows,
    representations,
    tml_models,
    gnn_models,
    repetitions,
    tml_samples=0,
    gnn_samples=0,
    tml_folds=5,
    gnn_folds=5,
    epochs=250,
):
    tml_fits = representations * tml_models * repetitions
    gnn_fits = gnn_models * repetitions
    final_fits = tml_fits + gnn_fits
    # Budget includes final refits. Actual searches may be capped by grid size,
    # reuse saved HPO results, or stop trials early.
    fit_budget = tml_fits * (1 + tml_samples * tml_folds) + gnn_fits * (1 + gnn_samples * gnn_folds)
    gnn_epochs = gnn_fits * (1 + gnn_samples * gnn_folds) * epochs
    reasons = []
    if final_fits >= 10:
        reasons.append(f"{final_fits:,} final model fits (threshold: 10)")
    if fit_budget >= 50:
        reasons.append(f"up to {fit_budget:,} fits including tuning/folds (threshold: 50)")
    if rows >= 5000 and (final_fits > 1 or gnn_fits):
        reasons.append(f"{rows:,} rows with multiple fits or GNN training (threshold: 5,000)")
    if gnn_epochs >= 2500:
        reasons.append(f"up to {gnn_epochs:,} GNN epochs across all fits (threshold: 2,500)")
    return Workload(final_fits, fit_budget, gnn_epochs, tuple(reasons))
