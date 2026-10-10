import streamlit as st

from polynet.app.options.state_keys import OptionalDependencyStateKeys
from polynet.utils.optional_dependencies import (
    PSMILES_CANONICALISER_LICENCE_URL,
    PSMILES_CANONICALISER_REQUIREMENT,
    install_psmiles_canonicaliser,
    psmiles_canonicaliser_available,
)


def require_psmiles_canonicaliser():
    """
    Stop the page until ``canonicalize-psmiles`` is installed, offering a
    one-click install after the user accepts its licence.
    """
    if psmiles_canonicaliser_available():
        return

    st.warning(
        "Your structures are PSMILES. PolyNet canonicalises PSMILES with the "
        "**canonicalize-psmiles** package (Kuenneth group), which is installed separately "
        "and is not installed yet."
    )
    st.markdown(
        "It is free for non-commercial use under the "
        f"[Georgia Tech licence]({PSMILES_CANONICALISER_LICENCE_URL}). "
        "Installing it means you accept that licence."
    )
    accepted = st.checkbox(
        "I accept the canonicalize-psmiles licence",
        key=OptionalDependencyStateKeys.AcceptPSMILESLicence,
    )
    if st.button(
        "Install canonicalize-psmiles",
        disabled=not accepted,
        key=OptionalDependencyStateKeys.InstallPSMILES,
    ):
        with st.spinner("Installing canonicalize-psmiles…"):
            result = install_psmiles_canonicaliser()
        if result.returncode == 0 and psmiles_canonicaliser_available():
            st.rerun()
        st.error(
            "The installation failed. You can install it from a terminal, in the same "
            "environment as PolyNet, with:"
        )
        st.code(f'pip install "{PSMILES_CANONICALISER_REQUIREMENT}"', language="bash")
        with st.expander("Installer output"):
            st.code(result.stdout + result.stderr)
    st.stop()
