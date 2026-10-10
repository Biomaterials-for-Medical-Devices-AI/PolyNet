"""
polynet.utils.optional_dependencies
===================================
Packages PolyNet uses but cannot declare as dependencies of the published
package, and how to install them.

``canonicalize-psmiles`` (Kuenneth group) canonicalises PSMILES strings. It is
not on PyPI, which rejects packages that depend on a URL, and its Georgia Tech
licence does not allow bundling it into MIT-licensed PolyNet. ``poetry install``
from a clone installs it (``psmiles`` dependency group); ``pip`` users install
it once with ``polynet install-psmiles`` or the button in the GUI.
"""

from __future__ import annotations

from collections.abc import Callable
import importlib
import importlib.util
import subprocess
import sys

PSMILES_CANONICALISER_MODULE = "canonicalize_psmiles"

# A pinned GitHub archive (same commit as poetry.lock) installs without git.
PSMILES_CANONICALISER_REQUIREMENT = (
    "canonicalize-psmiles @ https://github.com/kuennethgroup/canonicalize_psmiles/"
    "archive/42aebcf3e780198ba8bb50b50d4fa84dc8fb66fa.zip"
)

PSMILES_CANONICALISER_LICENCE_URL = (
    "https://github.com/kuennethgroup/canonicalize_psmiles/blob/main/LICENSE"
)

PSMILES_CANONICALISER_LICENCE_NOTICE = (
    "canonicalize-psmiles (Kuenneth group) is distributed by Georgia Tech Research "
    "Corporation under its own licence, which allows non-commercial use only. "
    f"Installing it means you accept that licence: {PSMILES_CANONICALISER_LICENCE_URL}"
)

PSMILES_CANONICALISER_MISSING = (
    "PSMILES canonicalisation needs the canonicalize-psmiles package, which is not "
    "installed. Install it once with:\n\n"
    "    polynet install-psmiles\n\n"
    "or use the install button in the PolyNet GUI."
)


def psmiles_canonicaliser_available() -> bool:
    """
    Return whether ``canonicalize-psmiles`` can be imported.

    Import caches are refreshed first, so a package installed while the
    process is running (e.g. from the GUI) is found.

    Returns:
        bool: True if the package is installed in this environment.
    """
    importlib.invalidate_caches()
    return importlib.util.find_spec(PSMILES_CANONICALISER_MODULE) is not None


def import_psmiles_canonicaliser() -> Callable[[str], str]:
    """
    Return the ``canonicalize`` function of ``canonicalize-psmiles``.

    Returns:
        Callable[[str], str]: Maps a PSMILES string to its canonical form.

    Raises:
        ImportError: If the package is not installed, explaining how to install it.
    """
    try:
        from canonicalize_psmiles.canonicalize import canonicalize
    except ImportError as e:
        raise ImportError(PSMILES_CANONICALISER_MISSING) from e
    return canonicalize


def install_psmiles_canonicaliser() -> subprocess.CompletedProcess:
    """
    Install ``canonicalize-psmiles`` into the running Python environment with pip.

    Returns:
        subprocess.CompletedProcess: The pip process, with its output captured
            as text; ``returncode`` 0 means it succeeded.
    """
    return subprocess.run(
        [sys.executable, "-m", "pip", "install", PSMILES_CANONICALISER_REQUIREMENT],
        capture_output=True,
        text=True,
    )
