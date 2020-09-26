from picturate.imports import *

__all__ = ["load_yaml"]


def load_yaml(path):
    with open(path, "r") as f:
        return edict(yaml.load(f, Loader=yaml.FullLoader))
