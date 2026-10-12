"""
tests/test_cli.py
=================
Streamlit options passed by ``polynet`` / ``polynet gui``.

The file watcher is off unless the user sets it: its module scan logs a
traceback for every ``transformers`` module needing a package PolyNet does not
install, which floods the terminal of a pip-installed GUI.
"""

import sys

import pytest

pytest.importorskip("streamlit")

from polynet import cli  # noqa: E402


@pytest.fixture
def streamlit_argv(monkeypatch):
    """Run ``polynet`` with the given arguments; return the ``streamlit`` argv."""
    from streamlit.web import cli as streamlit_cli

    monkeypatch.delenv(cli.FILE_WATCHER_ENV, raising=False)
    monkeypatch.setattr(streamlit_cli, "main", lambda: 0)

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["polynet", *args])
        cli.main()
        return sys.argv

    return run


def test_gui_starts_without_the_file_watcher(streamlit_argv):
    argv = streamlit_argv("gui", "--server.port", "8502")
    assert argv[:3] == ["streamlit", "run", str(cli.APP_ENTRYPOINT)]
    assert argv[3:] == ["--server.fileWatcherType", "none", "--server.port", "8502"]


@pytest.mark.parametrize(
    "args", [("--server.fileWatcherType", "auto"), ("--server.fileWatcherType=poll",)]
)
def test_a_file_watcher_set_by_the_user_is_kept(streamlit_argv, args):
    assert streamlit_argv(*args)[3:] == list(args)


def test_the_file_watcher_environment_variable_is_respected(streamlit_argv, monkeypatch):
    monkeypatch.setenv(cli.FILE_WATCHER_ENV, "auto")
    assert streamlit_argv()[3:] == []
