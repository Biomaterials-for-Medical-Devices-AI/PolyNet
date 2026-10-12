"""
tests/test_select_all.py
========================
"Select all" toggles must change the selection on every Streamlit version.
Recent versions keep a keyed multiselect's value when its ``default`` changes,
so toggles that only changed the default stopped working there.
"""

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402


def _app(options=("a", "b", "c"), select_all_default=False, use_checkbox=False):
    import streamlit as st

    from polynet.app.components.select_all import multiselect_with_select_all

    st.session_state["selected"] = multiselect_with_select_all(
        "Items",
        options=list(options),
        key="items",
        select_all_label="Select all",
        select_all_key="all",
        select_all_default=select_all_default,
        partial_selection=["a"],
        use_checkbox=use_checkbox,
    )


@pytest.mark.parametrize("use_checkbox", [False, True])
def test_switching_select_all_on_and_off_changes_the_selection(use_checkbox):
    app = AppTest.from_function(_app, kwargs={"use_checkbox": use_checkbox}).run()
    switch = app.checkbox[0] if use_checkbox else app.toggle[0]
    assert app.session_state["selected"] == ["a"]

    switch.set_value(True).run()
    assert app.session_state["selected"] == ["a", "b", "c"]

    switch = app.checkbox[0] if use_checkbox else app.toggle[0]
    switch.set_value(False).run()
    assert app.session_state["selected"] == ["a"]
    assert not app.exception and not app.warning


def test_starts_with_everything_selected_when_switched_on_by_default():
    app = AppTest.from_function(_app, kwargs={"select_all_default": True}).run()
    assert app.session_state["selected"] == ["a", "b", "c"]


def test_manual_edits_are_kept_on_later_reruns():
    app = AppTest.from_function(_app).run()
    app.toggle[0].set_value(True).run()
    app.multiselect[0].set_value(["b"]).run()
    app.run()
    assert app.session_state["selected"] == ["b"]


def test_options_that_disappear_are_dropped_from_the_selection():
    app = AppTest.from_function(_app, kwargs={"select_all_default": True}).run()
    app.kwargs["options"] = ("a", "c")  # e.g. another experiment
    app.run()
    assert app.session_state["selected"] == ["a", "c"]
    assert not app.exception
