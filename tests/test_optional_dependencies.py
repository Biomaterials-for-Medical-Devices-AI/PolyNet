"""
tests/test_optional_dependencies.py
===================================
canonicalize-psmiles is installed separately (not on PyPI). Without it, PSMILES
data must stop with install instructions, SMILES data must still work, and
``polynet install-psmiles`` installs it only after the licence is accepted.
"""

import subprocess
import sys

import pandas as pd
import pytest

from polynet import cli
from polynet.data.structures import prepare_structures
from polynet.utils import optional_dependencies
from polynet.utils.optional_dependencies import PSMILES_CANONICALISER_REQUIREMENT


@pytest.fixture
def without_canonicaliser(monkeypatch):
    # A None entry in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "canonicalize_psmiles", None)
    monkeypatch.setitem(sys.modules, "canonicalize_psmiles.canonicalize", None)
    monkeypatch.setattr(optional_dependencies, "psmiles_canonicaliser_available", lambda: False)


@pytest.fixture
def pip_calls(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(optional_dependencies.subprocess, "run", fake_run)
    return calls


def test_psmiles_without_canonicaliser_explains_how_to_install(without_canonicaliser):
    df = pd.DataFrame({"s": ["[*]CC[*]", "[*]OCC[*]"]})
    with pytest.raises(ImportError, match="polynet install-psmiles"):
        prepare_structures(df, ["s"])


def test_smiles_do_not_need_canonicaliser(without_canonicaliser):
    out, _ = prepare_structures(pd.DataFrame({"s": ["OCC"]}), ["s"])
    assert out["s"].tolist() == ["CCO"]


def test_install_declined_does_not_run_pip(without_canonicaliser, pip_calls, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    monkeypatch.setattr(sys, "argv", ["polynet", "install-psmiles"])
    assert cli.main() == 1
    assert pip_calls == []


def test_install_with_yes_runs_pip_in_this_environment(
    without_canonicaliser, pip_calls, monkeypatch
):
    monkeypatch.setattr(sys, "argv", ["polynet", "install-psmiles", "--yes"])
    assert cli.main() == 0
    assert pip_calls == [
        [sys.executable, "-m", "pip", "install", PSMILES_CANONICALISER_REQUIREMENT]
    ]
