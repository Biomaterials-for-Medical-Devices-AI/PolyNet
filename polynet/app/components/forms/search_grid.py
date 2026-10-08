"""
polynet.app.components.forms.search_grid
========================================
GUI widgets to customise hyperparameter search grids and the number of sampled
configurations (``hpo_search_grid`` / ``hpo_num_samples``).

Each tunable parameter is a multiselect pre-filled with its default
candidates; numeric parameters also accept extra values. Only the parameters
the user changes are returned, so an untouched editor reproduces the default
search exactly (same HPO cache key).
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from polynet.config.schemas.base import DEFAULT_HPO_NUM_SAMPLES


def _label(value: Any) -> str:
    """Display text of a parameter name or candidate value."""
    return "None" if value is None else str(getattr(value, "value", value))


def is_numeric_parameter(defaults: list) -> bool:
    """Whether a parameter takes numbers (its non-None defaults are all int/float, not bool)."""
    values = [v for v in defaults if v is not None]
    return bool(values) and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
    )


def parse_extra_values(text: str, defaults: list) -> list:
    """
    Parse comma-separated extra candidates for a numeric parameter.

    Values are integers when every default is an integer, floats otherwise.

    Parameters
    ----------
    text:
        User input, e.g. ``"0.2, 0.3"``.
    defaults:
        Default candidates of the parameter.

    Returns
    -------
    list
        The parsed values (empty for empty input).

    Raises
    ------
    ValueError
        If a value is not a number (or not an integer for integer parameters).
    """
    as_int = all(isinstance(v, int) for v in defaults if v is not None)
    values = []
    for item in (part.strip() for part in text.split(",")):
        if not item:
            continue
        number = float(item)
        if as_int:
            if not number.is_integer():
                raise ValueError(f"'{item}' is not a whole number.")
            number = int(number)
        values.append(number)
    return values


def changed_parameters(defaults: dict, chosen: dict) -> dict:
    """
    The parameters whose candidates differ from the defaults.

    Parameters
    ----------
    defaults:
        ``{parameter: default candidates}``.
    chosen:
        ``{parameter: candidates chosen by the user}``.

    Returns
    -------
    dict
        ``{parameter: candidates}`` for the changed parameters only.
    """
    return {p: values for p, values in chosen.items() if list(values) != list(defaults[p])}


def num_samples_widget(key: str, help_text: str) -> int:
    """Number of hyperparameter configurations sampled per HPO run."""
    return int(
        st.number_input(
            "Number of configurations to sample",
            min_value=1,
            value=DEFAULT_HPO_NUM_SAMPLES,
            step=10,
            key=key,
            help=help_text,
        )
    )


def search_grid_editor(defaults: dict, key_prefix: str, title: str) -> dict:
    """
    Expander to customise one search grid.

    Parameters
    ----------
    defaults:
        ``{parameter: default candidates}`` (e.g. ``default_gnn_shared_grid``).
    key_prefix:
        Unique prefix for the widget keys.
    title:
        Expander title.

    Returns
    -------
    dict
        The changed parameters, ready for ``hpo_search_grid`` (empty when
        nothing was changed).
    """
    if not defaults:
        return {}
    chosen: dict = {}
    with st.expander(title):
        st.caption(
            "Deselect candidates to exclude them; numeric parameters also accept extra "
            "values (comma-separated). Untouched parameters keep their default candidates."
        )
        for param, candidates in defaults.items():
            name = _label(param)
            selected = st.multiselect(
                name,
                options=candidates,
                default=candidates,
                format_func=_label,
                key=f"{key_prefix}{name}",
            )
            extra: list = []
            if is_numeric_parameter(candidates):
                text = st.text_input(
                    f"Extra values for {name}",
                    key=f"{key_prefix}{name}_extra",
                    placeholder="e.g. " + ", ".join(_label(v) for v in candidates[-2:]),
                )
                try:
                    extra = parse_extra_values(text, candidates)
                except ValueError as e:
                    st.error(f"{name}: {e} Use numbers separated by commas.")
                    st.stop()
            values = selected + [v for v in extra if v not in selected]
            if not values:
                st.error(f"Select at least one candidate for {name}.")
                st.stop()
            chosen[param] = values
    return changed_parameters(defaults, chosen)
