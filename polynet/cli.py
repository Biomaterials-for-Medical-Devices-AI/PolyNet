"""
polynet.cli
===========
Command-line entry point for PolyNet.

Installing the package exposes a ``polynet`` command, so both the GUI and
the YAML pipeline can be run from any working directory:

    polynet                          # launch the GUI
    polynet --server.port 8502       # any `streamlit run` option is forwarded
    polynet gui --server.port 8502   # same, explicit
    polynet run --config experiment.yaml   # run the pipeline from a YAML config
    polynet --version
"""

from pathlib import Path
import sys

from polynet import __version__

APP_ENTRYPOINT = Path(__file__).parent / "app" / "Welcome_to_PolyNet.py"


def launch_gui(args: list[str]) -> int:
    """
    Launch the PolyNet Streamlit GUI.

    Any arguments are forwarded to `streamlit run`, so Streamlit options such
    as `--server.port` work unchanged.

    Args:
        args (list[str]): Arguments forwarded to `streamlit run`.

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

    sys.argv = ["streamlit", "run", str(APP_ENTRYPOINT), *args]
    return streamlit_cli.main()


def main() -> int:
    """
    Dispatch ``polynet`` to the pipeline runner or the GUI.

    Returns:
        int: The process exit code.
    """
    args = sys.argv[1:]

    if args[:1] == ["--version"]:
        print(f"polynet {__version__}")
        return 0

    if args[:1] == ["run"]:
        from polynet.pipeline.runner import main as run_pipeline

        run_pipeline(args[1:])
        return 0

    if args[:1] == ["gui"]:
        args = args[1:]

    return launch_gui(args)


if __name__ == "__main__":
    sys.exit(main())
