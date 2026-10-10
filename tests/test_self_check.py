"""
tests/test_self_check.py
========================
``polynet check`` runs the real pipeline and must report a part as failed when
the pipeline logs an error and carries on (it does not raise), and pass when
every part runs.
"""

import logging

import pytest

from polynet.pipeline import runner, self_check


def _check(tmp_path, predict=False):
    return self_check._check_pipeline(
        "check", "regression", tmp_path, epochs=1, n_samples=20, predict=predict
    )


def test_error_logged_by_the_pipeline_fails_the_check(tmp_path, monkeypatch):
    # Without --verbose the console only shows critical messages.
    monkeypatch.setattr(logging.getLogger(), "level", logging.CRITICAL)

    def pipeline_with_failing_part(argv):
        logging.getLogger("polynet.pipeline").error("TML pipeline failed: boom")

    monkeypatch.setattr(runner, "main", pipeline_with_failing_part)
    result = _check(tmp_path)
    assert result.status == "FAIL"
    assert "TML pipeline failed: boom" in result.detail


def test_run_without_results_fails_the_check(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "main", lambda argv: None)
    result = _check(tmp_path)
    assert result.status == "FAIL"
    assert "metrics.json" in result.detail


@pytest.mark.integration
def test_full_check_passes(tmp_path):
    result = _check(tmp_path, predict=True)
    assert result.status == "PASS", result.detail
    out = tmp_path / "regression" / "results"
    assert (out / "explanations").is_dir()
    assert any((out / "unseen_predictions").rglob("predictions.csv"))
