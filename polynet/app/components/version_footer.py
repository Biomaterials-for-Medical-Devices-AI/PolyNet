"""
polynet.app.components.version_footer
=====================================
Version note pinned to the bottom of every GUI page.
"""

import streamlit as st

from polynet import __version__


def version_footer() -> None:
    """
    Show "You are using PolyNet vX.Y.Z" at the bottom of the window.

    Pinned with CSS rather than drawn at the end of the page, so that it also
    shows on pages that stop early (``st.stop()``); call it near the top of
    each page.
    """
    st.markdown(
        f"""
<div style="position: fixed; bottom: 0.5rem; right: 1rem; z-index: 1000;
            font-size: 0.9rem; opacity: 0.6; pointer-events: none;">
  You are using PolyNet v{__version__}
</div>
""",
        unsafe_allow_html=True,
    )
