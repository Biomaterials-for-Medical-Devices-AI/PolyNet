"""
polynet.app.components.select_all
=================================
A multiselect with a "select all" toggle or checkbox.

Recent Streamlit versions keep a keyed widget's value when its ``default``
changes, so a toggle that only changes the multiselect's ``default`` does
nothing there. Here the toggle sets the selection itself, through the session
state, when it is clicked.
"""

from collections.abc import Sequence

import streamlit as st


def multiselect_with_select_all(
    label: str,
    options: Sequence,
    key: str,
    *,
    select_all_label: str,
    select_all_key: str,
    select_all_default: bool = False,
    partial_selection: Sequence | None = None,
    use_checkbox: bool = False,
    select_all_help: str | None = None,
    **multiselect_kwargs,
) -> list:
    """
    Render a "select all" toggle (or checkbox) above a multiselect.

    Switching it on selects every option; switching it off selects
    ``partial_selection`` (nothing by default). The user can still edit the
    selection afterwards.

    Args:
        label: Label of the multiselect.
        options: The multiselect options.
        key: Session-state key of the multiselect.
        select_all_label: Label of the toggle.
        select_all_key: Session-state key of the toggle.
        select_all_default: Whether the toggle starts switched on.
        partial_selection: Selection while the toggle is off.
        use_checkbox: Show a checkbox instead of a toggle.
        select_all_help: Help text of the toggle.
        **multiselect_kwargs: Passed to ``st.multiselect`` (e.g. ``help``).

    Returns:
        The selected options.
    """
    options = list(options)
    partial = [o for o in (partial_selection or []) if o in options]

    def selection(select_all: bool) -> list:
        return list(options) if select_all else list(partial)

    def apply_select_all() -> None:
        st.session_state[key] = selection(st.session_state[select_all_key])

    (st.checkbox if use_checkbox else st.toggle)(
        select_all_label,
        value=select_all_default,
        key=select_all_key,
        help=select_all_help,
        on_change=apply_select_all,
    )
    if key not in st.session_state:
        st.session_state[key] = selection(st.session_state[select_all_key])
    else:
        # Options can change between runs (e.g. another experiment).
        st.session_state[key] = [o for o in st.session_state[key] if o in options]
    return st.multiselect(label, options=options, key=key, **multiselect_kwargs)
