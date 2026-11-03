"""[new] One config.yaml per language instead of the reference's three yaml files.

The file has three top-level sections (preprocess / model / train) that are handed to
the reference code unchanged, as the (preprocess_config, model_config, train_config)
tuple it already expects. Relative paths are resolved against the config's folder so
the scripts work from any working directory.
"""

import os

import yaml


def _resolve_paths(section, base):
    for key, value in section.items():
        if isinstance(value, str) and not os.path.isabs(value):
            section[key] = os.path.normpath(os.path.join(base, value))


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(path))
    _resolve_paths(config["preprocess"]["path"], base)
    _resolve_paths(config["train"]["path"], base)
    return config["preprocess"], config["model"], config["train"]
