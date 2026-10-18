"""
polynet.app.components.images
=============================
Images that fill the width of their container, on every supported Streamlit.

Streamlit 1.5x replaced ``use_container_width`` with ``width`` and prints a
deprecation notice for the old name on every call, while the older versions
PolyNet supports only understand ``use_container_width``. The other elements
(plots, dataframes) fill the container by default in both, so only images need
to say so.
"""

import inspect

import streamlit as st

# ``width`` takes "stretch" where its default is a string ("content"); where it
# is a pixel count, this Streamlit only knows ``use_container_width``.
_FULL_WIDTH = (
    {"width": "stretch"}
    if isinstance(inspect.signature(st.image).parameters["width"].default, str)
    else {"use_container_width": True}
)


def show_image_full_width(image, container=st, **options) -> None:
    """Show ``image`` filling the width of ``container``."""
    container.image(image, **options, **_FULL_WIDTH)
