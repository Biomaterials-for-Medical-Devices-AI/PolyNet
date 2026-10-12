"""
tests/test_version_footer.py
============================
Every GUI page tells the user which PolyNet version they are running, also
pages that stop early (no experiment yet).
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from polynet import __version__  # noqa: E402

_APP = Path(__file__).parents[1] / "polynet" / "app"
_PAGES = [_APP / "Welcome_to_PolyNet.py", *sorted((_APP / "pages").glob("[0-9]_*.py"))]


@pytest.mark.parametrize("page", _PAGES, ids=lambda p: p.stem)
def test_every_page_shows_the_polynet_version(page, tmp_path, monkeypatch):
    # No experiments: the pages stop early, after the footer.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    app = AppTest.from_file(str(page), default_timeout=60).run()
    assert any(f"You are using PolyNet v{__version__}" in m.value for m in app.markdown)
