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
    polynet check                    # check that the installation works
    polynet install-psmiles          # install the PSMILES canonicaliser (once)
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


def install_psmiles(args: list[str]) -> int:
    """
    Install ``canonicalize-psmiles``, which PolyNet needs for PSMILES data.

    Shows its licence and asks for confirmation unless ``--yes`` is given.

    Args:
        args (list[str]): ``["--yes"]`` to skip the confirmation prompt.

    Returns:
        int: 0 if the package is installed, otherwise 1.
    """
    from polynet.utils.optional_dependencies import (
        PSMILES_CANONICALISER_LICENCE_NOTICE,
        install_psmiles_canonicaliser,
        psmiles_canonicaliser_available,
    )

    if psmiles_canonicaliser_available():
        print("canonicalize-psmiles is already installed.")
        return 0

    print(PSMILES_CANONICALISER_LICENCE_NOTICE)
    if "--yes" not in args and input("Install it now? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Not installed.")
        return 1

    result = install_psmiles_canonicaliser()
    if result.returncode != 0:
        print(result.stdout, result.stderr, sep="\n", file=sys.stderr)
        return 1
    print("canonicalize-psmiles installed.")
    return 0


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

    if args[:1] == ["check"]:
        from polynet.pipeline.self_check import main as self_check

        return self_check(args[1:])

    if args[:1] == ["install-psmiles"]:
        return install_psmiles(args[1:])

    if args[:1] == ["gui"]:
        args = args[1:]

    return launch_gui(args)


if __name__ == "__main__":
    sys.exit(main())
