from picturate.imports import *
from picturate.config.util import *

__all__ = ["CAttnGANConfig"]

def CAttnGANConfig(filename):

    yaml_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'yaml_files')

    if filename == 'bird':
        path = os.path.join(yaml_dir, "{}_cycle.yaml".format(filename))

    return load_yaml(path)

