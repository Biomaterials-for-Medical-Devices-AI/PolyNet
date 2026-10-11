from polynet.app.services.cache import read_json
from polynet.config.io import save_options


def load_options(path, options_class):
    return options_class(**read_json(path))


__all__ = ["load_options", "save_options"]
