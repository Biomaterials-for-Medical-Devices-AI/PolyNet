"""
Run the full PolyNet pipeline from a YAML config.

Kept for existing workflows; equivalent to the installed command::

    polynet run --config configs/experiment.yaml

See ``polynet.pipeline.runner`` for all options.
"""

from polynet.pipeline.runner import main

if __name__ == "__main__":
    main()
