"""
polynet.cli
===========
Command-line entry point for the PolyNet Streamlit GUI.

Installing the package exposes a ``polynet`` command, so the app can be
launched from any working directory without knowing the path to the
Streamlit script:

    polynet                          # launch the GUI
    polynet --server.port 8502       # any `streamlit run` option is forwarded
"""

from pathlib import Path
import sys

APP_ENTRYPOINT = Path(__file__).parent / "app" / "Welcome_to_PolyNet.py"


def main() -> int:
    """
    Launch the PolyNet Streamlit GUI.

    Any arguments given on the command line are forwarded to `streamlit run`,
    so Streamlit options such as `--server.port` work unchanged.

    Returns:
        int: A non-zero exit code if Streamlit is not installed. Otherwise
            Streamlit itself terminates the process when the server stops.
    """
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError:
        print(
            "The PolyNet GUI requires Streamlit, which is not installed in this "
            "environment.\nInstall it with:\n\n    pip install streamlit\n",
            file=sys.stderr,
        )
        return 1

    sys.argv = ["streamlit", "run", str(APP_ENTRYPOINT), *sys.argv[1:]]
    return streamlit_cli.main()


if __name__ == "__main__":
    sys.exit(main())
